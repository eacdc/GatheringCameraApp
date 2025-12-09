import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from tkinter import font as tkfont
import cv2
import numpy as np
import pytesseract
from pathlib import Path
from rapidfuzz import fuzz
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime

# ================== CONFIG ==================
DEFAULT_TOLERANCE = 80
LANGS = "eng"
MASTER_TEXT_FILE = Path("master_text.txt")
MIN_OCR_CONFIDENCE = 50
MIN_OCR_TEXT_LEN = 30

# Windows Tesseract path (uncomment if needed)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ================== IMAGE / OCR UTILITIES ==================
def measure_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def is_paper_plausible(rect):
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    w = max(widthA, widthB)
    h = max(heightA, heightB)
    if w < 100 or h < 150:
        return False
    aspect = w / h if h != 0 else 0
    if aspect < 0.6 or aspect > 2.0:
        return False
    return True


def find_paper_and_warp(image):
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
    margin = int(0.02 * min(maxWidth, maxHeight))
    if margin > 0 and warped.shape[0] > 2 * margin and warped.shape[1] > 2 * margin:
        warped = warped[margin:-margin, margin:-margin]
    return warped


def ocr_with_confidence(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 1.5
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
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


# ================== DESKTOP GUI APPLICATION ==================
class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image OCR Testing Tool - Desktop")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)
        
        # State
        self.current_image = None
        self.current_processed = None
        self.current_ocr_text = None
        self.current_confidence = None
        self.master_text = load_master_text()
        
        self.setup_ui()
        
        # Load master text if exists
        if self.master_text:
            self.master_text_area.insert("1.0", self.master_text)
    
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Image OCR Testing Tool",
            font=tkfont.Font(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Left panel - Image and OCR
        left_frame = ttk.LabelFrame(main_frame, text="Image & OCR", padding="10")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=1)
        
        # Image selection buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(
            button_frame,
            text="📁 Select Image",
            command=self.select_image
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            button_frame,
            text="🔄 Process OCR",
            command=self.process_image,
            state="disabled"
        ).pack(side=tk.LEFT, padx=(0, 5))
        self.process_btn = button_frame.winfo_children()[1]
        
        ttk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self.clear_image
        ).pack(side=tk.LEFT)
        
        # Image preview
        self.image_label = ttk.Label(left_frame, text="No image selected", anchor="center")
        self.image_label.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # OCR Results
        ocr_frame = ttk.LabelFrame(left_frame, text="OCR Results", padding="10")
        ocr_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        ocr_frame.columnconfigure(0, weight=1)
        ocr_frame.rowconfigure(1, weight=1)
        
        # OCR metrics
        metrics_frame = ttk.Frame(ocr_frame)
        metrics_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.confidence_label = ttk.Label(metrics_frame, text="Confidence: -")
        self.confidence_label.pack(side=tk.LEFT, padx=(0, 15))
        
        self.status_label = ttk.Label(metrics_frame, text="Status: -")
        self.status_label.pack(side=tk.LEFT)
        
        # OCR text display
        self.ocr_text_area = scrolledtext.ScrolledText(
            ocr_frame,
            height=8,
            wrap=tk.WORD,
            font=("Consolas", 10)
        )
        self.ocr_text_area.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Right panel - Comparison
        right_frame = ttk.LabelFrame(main_frame, text="Text Comparison", padding="10")
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        right_frame.rowconfigure(3, weight=1)
        
        # Master text
        ttk.Label(right_frame, text="Master Text (Reference):").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.master_text_area = scrolledtext.ScrolledText(
            right_frame,
            height=8,
            wrap=tk.WORD,
            font=("Consolas", 10)
        )
        self.master_text_area.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Master text buttons
        master_btn_frame = ttk.Frame(right_frame)
        master_btn_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(
            master_btn_frame,
            text="💾 Save Master Text",
            command=self.save_master
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            master_btn_frame,
            text="📋 Load from OCR",
            command=self.load_from_ocr,
            state="disabled"
        ).pack(side=tk.LEFT)
        self.load_ocr_btn = master_btn_frame.winfo_children()[1]
        
        # Comparison
        ttk.Label(right_frame, text="Comparison Result:").grid(row=3, column=0, sticky=tk.W, pady=(0, 5))
        
        comparison_frame = ttk.Frame(right_frame)
        comparison_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        comparison_frame.columnconfigure(0, weight=1)
        
        self.similarity_label = ttk.Label(
            comparison_frame,
            text="Similarity: -",
            font=tkfont.Font(size=14, weight="bold")
        )
        self.similarity_label.grid(row=0, column=0, pady=5)
        
        self.match_status_label = ttk.Label(
            comparison_frame,
            text="",
            font=tkfont.Font(size=12)
        )
        self.match_status_label.grid(row=1, column=0)
        
        ttk.Button(
            comparison_frame,
            text="🔍 Compare Texts",
            command=self.compare_texts
        ).grid(row=2, column=0, pady=(10, 0))
        self.compare_btn = comparison_frame.winfo_children()[2]
        
        # Status bar
        self.status_bar = ttk.Label(
            main_frame,
            text="Ready",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        try:
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Error", "Failed to load image file")
                return
            
            self.current_image = image
            self.display_image(image)
            self.process_btn.config(state="normal")
            self.status_bar.config(text=f"Image loaded: {Path(file_path).name}")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")
    
    def display_image(self, image, max_size=(600, 400)):
        # Resize for display
        h, w = image.shape[:2]
        scale = min(max_size[0] / w, max_size[1] / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        
        if scale < 1:
            resized = cv2.resize(image, (new_w, new_h))
        else:
            resized = image
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        photo = ImageTk.PhotoImage(image=pil_image)
        
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo  # Keep a reference
    
    def process_image(self):
        if self.current_image is None:
            return
        
        self.status_bar.config(text="Processing image...")
        self.process_btn.config(state="disabled")
        
        # Process in thread to avoid freezing UI
        thread = threading.Thread(target=self._process_image_thread)
        thread.daemon = True
        thread.start()
    
    def _process_image_thread(self):
        try:
            # Try to find and warp paper
            processed = find_paper_and_warp(self.current_image)
            if processed is None:
                processed = self.current_image.copy()
            
            self.current_processed = processed
            
            # Run OCR
            text, confidence = ocr_with_confidence(processed)
            self.current_ocr_text = text
            self.current_confidence = confidence
            
            # Update UI in main thread
            self.root.after(0, self._update_ocr_results, text, confidence, processed)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"OCR Error: {str(e)}"))
            self.root.after(0, lambda: self.status_bar.config(text="Error processing image"))
            self.root.after(0, lambda: self.process_btn.config(state="normal"))
    
    def _update_ocr_results(self, text, confidence, processed_image):
        # Update OCR text
        self.ocr_text_area.delete("1.0", tk.END)
        self.ocr_text_area.insert("1.0", text if text else "No text detected")
        
        # Update metrics
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
        
        # Determine status
        if confidence < MIN_OCR_CONFIDENCE:
            status = "LOW_CONFIDENCE"
            status_color = "red"
        elif len(text) < MIN_OCR_TEXT_LEN:
            status = "SHORT_TEXT"
            status_color = "orange"
        elif not text.strip():
            status = "NO_TEXT"
            status_color = "red"
        else:
            status = "OK"
            status_color = "green"
        
        self.status_label.config(text=f"Status: {status}", foreground=status_color)
        
        # Display processed image
        if processed_image is not None:
            self.display_image(processed_image)
        
        # Enable buttons
        self.load_ocr_btn.config(state="normal" if text else "disabled")
        self.compare_btn.config(state="normal" if text and self.master_text_area.get("1.0", tk.END).strip() else "disabled")
        self.process_btn.config(state="normal")
        
        self.status_bar.config(text=f"OCR completed - Confidence: {confidence:.1f}%")
    
    def clear_image(self):
        self.current_image = None
        self.current_processed = None
        self.current_ocr_text = None
        self.current_confidence = None
        self.image_label.config(image="", text="No image selected")
        self.image_label.image = None
        self.ocr_text_area.delete("1.0", tk.END)
        self.confidence_label.config(text="Confidence: -")
        self.status_label.config(text="Status: -", foreground="black")
        self.process_btn.config(state="disabled")
        self.load_ocr_btn.config(state="disabled")
        self.compare_btn.config(state="disabled")
        self.status_bar.config(text="Cleared")
    
    def save_master(self):
        text = self.master_text_area.get("1.0", tk.END).strip()
        if text:
            save_master_text(text)
            self.master_text = text
            messagebox.showinfo("Success", "Master text saved successfully!")
            self.compare_btn.config(state="normal" if self.current_ocr_text else "disabled")
        else:
            messagebox.showwarning("Warning", "Master text is empty")
    
    def load_from_ocr(self):
        if self.current_ocr_text:
            self.master_text_area.delete("1.0", tk.END)
            self.master_text_area.insert("1.0", self.current_ocr_text)
            self.save_master()
        else:
            messagebox.showwarning("Warning", "No OCR text available")
    
    def compare_texts(self):
        master = self.master_text_area.get("1.0", tk.END).strip()
        current = self.current_ocr_text
        
        if not master:
            messagebox.showwarning("Warning", "Please enter master text")
            return
        
        if not current:
            messagebox.showwarning("Warning", "No OCR text to compare")
            return
        
        similarity = compare_text(master, current)
        
        # Update similarity display
        self.similarity_label.config(text=f"Similarity: {similarity:.1f}%")
        
        # Update match status
        if similarity >= 80:
            status_text = "✓ MATCH"
            status_color = "green"
        elif similarity >= 50:
            status_text = "⚠ PARTIAL MATCH"
            status_color = "orange"
        else:
            status_text = "✗ NO MATCH"
            status_color = "red"
        
        self.match_status_label.config(text=status_text, foreground=status_color)
        self.status_bar.config(text=f"Comparison complete - Similarity: {similarity:.1f}%")


# ================== ENTRY POINT ==================
if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()

