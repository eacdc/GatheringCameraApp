import time
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from rapidfuzz import fuzz
import os

# ================== CONFIG ==================
CAPTURE_INTERVAL_SEC = 5         # process image every 5 seconds (faster for testing)
DEFAULT_TOLERANCE = 80           # % similarity match
LANGS = "eng"                    # English only for now

# Image folder to read from
TEST_IMAGES_FOLDER = Path("test_images")  # Put your test images in this folder

MASTER_TEXT_FILE = Path("master_text.txt")
LATEST_IMAGE_PATH = Path("static/latest_paper.jpg")
TEST_UPLOAD_FOLDER = Path("static/test_uploads")
TEST_PROCESSED_FOLDER = Path("static/test_processed")

# Create folders if they don't exist
TEST_UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
TEST_PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
Path("static").mkdir(parents=True, exist_ok=True)

# Quality / filtering settings
MIN_OCR_CONFIDENCE = 50          # minimum average OCR confidence to accept
MIN_OCR_TEXT_LEN = 30            # minimum characters of OCR text to accept

SKIN_RATIO_THRESHOLD = 0.02      # >2% skin-like pixels => assume hand present
MISMATCH_STREAK_THRESHOLD = 3    # require 3 consecutive mismatches before going RED

# Windows Tesseract path (uncomment and set if needed)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


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
    "current_image_index": 0,   # track which image we're processing
}
state_lock = threading.Lock()


# ================== IMAGE READING FROM FOLDER ==================
def get_image_files():
    """Get all image files from the test_images folder."""
    if not TEST_IMAGES_FOLDER.exists():
        TEST_IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
        return []
    
    # Supported image formats
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    for ext in extensions:
        image_files.extend(TEST_IMAGES_FOLDER.glob(ext))
        image_files.extend(TEST_IMAGES_FOLDER.glob(ext.upper()))
    
    # Sort for consistent ordering
    return sorted(image_files)


def load_image_from_folder():
    """
    Load next image from test_images folder in a loop.
    Returns a BGR image (OpenCV format) or None if no images found.
    """
    image_files = get_image_files()
    if not image_files:
        return None
    
    with state_lock:
        index = state["current_image_index"]
        state["current_image_index"] = (index + 1) % len(image_files)
        image_path = image_files[index]
    
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    return img


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


# ================== BACKGROUND IMAGE PROCESSING LOOP ==================
def capture_loop():
    with state_lock:
        state["master_text"] = load_master_text()

    # Check if test_images folder exists and has images
    image_files = get_image_files()
    if not image_files:
        with state_lock:
            state["last_status"] = "ERROR"
            state["last_error"] = f"No images found in '{TEST_IMAGES_FOLDER}' folder. Please add some test images."
            state["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"ERROR: No images found in '{TEST_IMAGES_FOLDER}' folder.")
        print(f"Please create the folder and add some test images (jpg, png, etc.)")
        return

    print(f"Found {len(image_files)} image(s) in '{TEST_IMAGES_FOLDER}' folder. Starting processing loop...")

    try:
        while True:
            # Load next image from folder
            image = load_image_from_folder()
            
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if image is None:
                with state_lock:
                    state["last_status"] = "ERROR"
                    state["last_error"] = "Failed to load image from folder."
                    state["last_updated"] = now_str
                time.sleep(CAPTURE_INTERVAL_SEC)
                continue

            # Hand / finger detection – skip this cycle if hand is present
            if has_hand_like_region(image):
                with state_lock:
                    state["last_status"] = "NO_PAPER"
                    state["last_error"] = "Hand detected, skipping this frame."
                    state["last_updated"] = now_str
                time.sleep(CAPTURE_INTERVAL_SEC)
                continue

            paper = find_paper_and_warp(image)
            if paper is None:
                # If paper detection fails, try using the whole image directly
                paper = image.copy()
                with state_lock:
                    state["last_status"] = "NO_PAPER"
                    state["last_error"] = "Paper not detected, using full image for OCR."
                    state["last_updated"] = now_str

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

    except KeyboardInterrupt:
        print("\nStopping image processing loop...")


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


# ================== TEST PAGE ROUTES ==================
@app.route("/test", methods=["GET"])
def test_page():
    """Render the test page for Windows testing."""
    return render_template("test.html")


@app.route("/test/process", methods=["POST"])
def test_process_image():
    """Process uploaded image and return OCR results."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        upload_path = TEST_UPLOAD_FOLDER / f"{timestamp}_{filename}"
        file.save(str(upload_path))
        
        # Read image
        image = cv2.imread(str(upload_path))
        if image is None:
            return jsonify({"error": "Failed to read image file"}), 400
        
        # Try to find and warp paper
        paper = find_paper_and_warp(image)
        processed_image_path = None
        
        if paper is not None:
            # Save processed paper image
            processed_filename = f"processed_{timestamp}_{filename}"
            processed_image_path = TEST_PROCESSED_FOLDER / processed_filename
            cv2.imwrite(str(processed_image_path), paper)
            ocr_image = paper
        else:
            # Use full image if paper detection fails
            ocr_image = image
        
        # Run OCR
        try:
            text, avg_conf = ocr_with_confidence(ocr_image)
        except Exception as e:
            return jsonify({"error": f"OCR error: {str(e)}"}), 500
        
        # Determine status
        status = "OK"
        if avg_conf < MIN_OCR_CONFIDENCE:
            status = "LOW_CONFIDENCE"
        if len(text) < MIN_OCR_TEXT_LEN:
            status = "SHORT_TEXT"
        if not text.strip():
            status = "NO_TEXT"
        
        # Prepare response
        response = {
            "text": text,
            "confidence": avg_conf,
            "status": status,
            "text_length": len(text)
        }
        
        # Add processed image URL if available
        if processed_image_path:
            response["processed_image_url"] = url_for('static', filename=f'test_processed/{processed_filename}')
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


@app.route("/test/compare", methods=["POST"])
def test_compare_texts():
    """Compare two texts and return similarity score."""
    try:
        data = request.get_json()
        master = data.get("master", "").strip()
        current = data.get("current", "").strip()
        
        if not master or not current:
            return jsonify({"error": "Both master and current text are required"}), 400
        
        similarity = compare_text(master, current)
        
        return jsonify({
            "similarity": similarity,
            "master": master,
            "current": current
        })
    
    except Exception as e:
        return jsonify({"error": f"Comparison error: {str(e)}"}), 500


@app.route("/test/get_master", methods=["GET"])
def test_get_master():
    """Get current master text."""
    with state_lock:
        master_text = state.get("master_text")
    
    return jsonify({
        "master_text": master_text or ""
    })


# ================== ENTRY POINT ==================
if __name__ == "__main__":
    worker = threading.Thread(target=capture_loop, daemon=True)
    worker.start()

    print("\n" + "="*60)
    print("Windows Test Version - Image Processing App")
    print("="*60)
    print(f"Put your test images in the '{TEST_IMAGES_FOLDER}' folder")
    print("Supported formats: jpg, jpeg, png, bmp, tiff")
    print(f"Processing interval: {CAPTURE_INTERVAL_SEC} seconds")
    print("\nStarting Flask server on http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=False)

