import time
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from flask import Flask, render_template, request, redirect, url_for
from picamera2 import Picamera2
from rapidfuzz import fuzz

# ================== CONFIG ==================
CAPTURE_INTERVAL_SEC = 60        # capture every 60 seconds
FRAMES_PER_CYCLE = 4             # capture N frames, pick sharpest
DEFAULT_TOLERANCE = 80           # % similarity match
LANGS = "eng"                    # English only for now

MASTER_TEXT_FILE = Path("master_text.txt")
LATEST_IMAGE_PATH = Path("static/latest_paper.jpg")

# Quality / filtering settings
MIN_OCR_CONFIDENCE = 50          # minimum average OCR confidence to accept
MIN_OCR_TEXT_LEN = 30            # minimum characters of OCR text to accept

SKIN_RATIO_THRESHOLD = 0.02      # >2% skin-like pixels => assume hand present
MISMATCH_STREAK_THRESHOLD = 3    # require 3 consecutive mismatches before going RED

# If needed, explicitly set Tesseract path (usually not needed on Pi OS):
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


# ================== GLOBAL STATE ==================
state = {
    "master_text": None,
    "tolerance": DEFAULT_TOLERANCE,
    "last_text": "",
    "last_score": None,
    "last_status": "INIT",      # INIT / NO_MASTER / OK / MISMATCH / ERROR / NO_PAPER
    "last_updated": None,
    "last_error": "",
    "image_timestamp": None,    # for cache-busting in the UI
    "mismatch_streak": 0,
}
state_lock = threading.Lock()


# ================== CAMERA (PICAMERA2) ==================
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)})
picam2.configure(camera_config)
picam2.start()


def capture_frame():
    """
    Capture one frame from Pi Camera v3 via Picamera2.
    Returns a BGR image (OpenCV format).
    """
    rgb = picam2.capture_array()               # RGB image
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) # convert to BGR
    return bgr


# ================== IMAGE / OCR UTILITIES ==================
def measure_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def is_paper_plausible(rect):
    """
    Basic sanity checks on paper quadrilateral (size & aspect ratio).
    """
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    w = max(widthA, widthB)
    h = max(heightA, heightB)

    # Too small to be your sheet
    if w < 100 or h < 150:
        return False

    aspect = w / h if h != 0 else 0
    # Rough A4-ish bounds (adjust if needed)
    if aspect < 0.6 or aspect > 2.0:
        return False

    return True


def find_paper_and_warp(image):
    """
    Detect the largest 4-sided contour (paper) and apply perspective warp.
    Returns cropped paper image or None if not found / not plausible.
    """
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    paper_contour = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            paper_contour = approx
            break

    if paper_contour is None:
        return None

    pts = paper_contour.reshape(4, 2)
    rect = order_points(pts)

    if not is_paper_plausible(rect):
        return None

    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    if maxWidth < 50 or maxHeight < 50:
        return None

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

    # Small margin crop to remove borders
    margin = int(0.02 * min(maxWidth, maxHeight))
    if margin > 0 and warped.shape[0] > 2 * margin and warped.shape[1] > 2 * margin:
        warped = warped[margin:-margin, margin:-margin]

    return warped


def has_hand_like_region(image):
    """
    Crude hand/finger detection using a simple skin-color range in HSV.
    If skin-colored pixels exceed SKIN_RATIO_THRESHOLD, assume hand present.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # These bounds may need tuning for your lighting/skin tones
    lower = np.array([0, 30, 60])
    upper = np.array([20, 200, 255])
    mask = cv2.inRange(hsv, lower, upper)
    skin_pixels = cv2.countNonZero(mask)
    total_pixels = image.shape[0] * image.shape[1]
    ratio = skin_pixels / total_pixels
    return ratio > SKIN_RATIO_THRESHOLD


def ocr_with_confidence(image):
    """
    Run Tesseract OCR and return (text, avg_confidence).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Upscale to improve OCR for smaller text
    scale = 1.5
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Binarize
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    data = pytesseract.image_to_data(
        thresh,
        lang=LANGS,
        config="--psm 3",
        output_type=pytesseract.Output.DICT,
    )

    words = [w for w in data["text"] if w.strip()]
    text = " ".join(words)

    confs = [int(c) for c in data["conf"] if c != "-1"]
    avg_conf = sum(confs) / len(confs) if confs else 0

    return text, avg_conf


def compare_text(master, current):
    def norm(t):
        return " ".join(t.lower().split())
    return fuzz.ratio(norm(master), norm(current))


def load_master_text():
    if MASTER_TEXT_FILE.exists():
        return MASTER_TEXT_FILE.read_text(encoding="utf-8")
    return None


def save_master_text(text):
    MASTER_TEXT_FILE.write_text(text, encoding="utf-8")


# ================== BACKGROUND CAPTURE / OCR LOOP ==================
def capture_loop():
    with state_lock:
        state["master_text"] = load_master_text()

    try:
        while True:
            sharpest_image = None
            best_sharpness = -1

            # Capture multiple frames and pick the sharpest
            for _ in range(FRAMES_PER_CYCLE):
                frame = capture_frame()
                sharpness = measure_sharpness(frame)
                if sharpness > best_sharpness:
                    best_sharpness = sharpness
                    sharpest_image = frame
                time.sleep(0.2)

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if sharpest_image is None:
                with state_lock:
                    state["last_status"] = "ERROR"
                    state["last_error"] = "Failed to capture any frame."
                    state["last_updated"] = now_str
                time.sleep(CAPTURE_INTERVAL_SEC)
                continue

            # Hand / finger detection – skip this cycle if hand is present
            if has_hand_like_region(sharpest_image):
                with state_lock:
                    state["last_status"] = "NO_PAPER"
                    state["last_error"] = "Hand detected, skipping this frame."
                    state["last_updated"] = now_str
                time.sleep(CAPTURE_INTERVAL_SEC)
                continue

            paper = find_paper_and_warp(sharpest_image)
            if paper is None:
                with state_lock:
                    state["last_status"] = "NO_PAPER"
                    state["last_error"] = "Paper not detected or shape not plausible."
                    state["last_updated"] = now_str
                time.sleep(CAPTURE_INTERVAL_SEC)
                continue

            # Save latest paper image for UI
            LATEST_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(LATEST_IMAGE_PATH), paper)

            # OCR with confidence
            try:
                text, avg_conf = ocr_with_confidence(paper)
            except Exception as e:
                with state_lock:
                    state["last_status"] = "ERROR"
                    state["last_error"] = f"OCR error: {e}"
                    state["last_updated"] = now_str
                time.sleep(CAPTURE_INTERVAL_SEC)
                continue

            # Quality gate: skip very low confidence or too-short text
            if avg_conf < MIN_OCR_CONFIDENCE or len(text) < MIN_OCR_TEXT_LEN:
                with state_lock:
                    state["last_status"] = "NO_PAPER"
                    state["last_error"] = (
                        f"OCR low confidence ({avg_conf:.1f}) or text too short "
                        f"({len(text)} chars), skipping this cycle."
                    )
                    state["last_updated"] = now_str
                time.sleep(CAPTURE_INTERVAL_SEC)
                continue

            # At this point, we have good OCR; update and compare
            with state_lock:
                state["last_text"] = text
                state["last_updated"] = now_str
                state["image_timestamp"] = int(time.time())

                master = state["master_text"]
                tolerance = state["tolerance"]

                if not master:
                    state["last_status"] = "NO_MASTER"
                    state["last_score"] = None
                    state["last_error"] = "Master text not set."
                else:
                    score = compare_text(master, text)
                    state["last_score"] = score

                    if score >= tolerance:
                        state["mismatch_streak"] = 0
                        state["last_status"] = "OK"
                        state["last_error"] = ""
                    else:
                        state["mismatch_streak"] += 1
                        if state["mismatch_streak"] >= MISMATCH_STREAK_THRESHOLD:
                            state["last_status"] = "MISMATCH"
                            state["last_error"] = ""
                        else:
                            # Treat as temporary mismatch; don't go RED yet
                            state["last_status"] = "NO_PAPER"
                            state["last_error"] = (
                                "Temporary mismatch, waiting for confirmation "
                                f"({state['mismatch_streak']}/"
                                f"{MISMATCH_STREAK_THRESHOLD})."
                            )

            time.sleep(CAPTURE_INTERVAL_SEC)

    finally:
        picam2.stop()


# ================== FLASK APP ==================
app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    with state_lock:
        context = dict(state)
    return render_template("index.html", **context)


@app.route("/set_master", methods=["POST"])
def set_master():
    use_last = request.form.get("source", "last") == "last"
    text_from_form = request.form.get("master_text", "").strip()

    with state_lock:
        if use_last:
            new_master = state["last_text"] or ""
        else:
            new_master = text_from_form

        if new_master:
            state["master_text"] = new_master
            save_master_text(new_master)
            # Reset mismatch streak when master changes
            state["mismatch_streak"] = 0

    return redirect(url_for("index"))


@app.route("/update_tolerance", methods=["POST"])
def update_tolerance():
    try:
        t = int(request.form.get("tolerance", DEFAULT_TOLERANCE))
        t = max(0, min(100, t))
    except ValueError:
        t = DEFAULT_TOLERANCE

    with state_lock:
        state["tolerance"] = t

    return redirect(url_for("index"))


# ================== ENTRY POINT ==================
if __name__ == "__main__":
    worker = threading.Thread(target=capture_loop, daemon=True)
    worker.start()

    app.run(host="0.0.0.0", port=5000, debug=False)
