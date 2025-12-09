# Windows Test Version - Setup Instructions

This is a Windows-compatible version of the image processing app that reads images from a folder instead of using a camera.

## Installation

### 1. Install Tesseract OCR

Download and install Tesseract OCR for Windows:
- **Download**: https://github.com/UB-Mannheim/tesseract/wiki
- **Or use Chocolatey**: `choco install tesseract`

After installation, if Tesseract is not in your PATH, you may need to uncomment and set the path in `app_windows.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### 2. Install Python Dependencies

```bash
pip install -r requirements_windows.txt
```

Or using a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements_windows.txt
```

## Usage

### 1. Create Test Images Folder

Create a folder named `test_images` in the same directory as `app_windows.py`:
```
GatheringCameraApp/
├── app_windows.py
├── test_images/          ← Create this folder
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
```

### 2. Add Your Test Images

Place your test images in the `test_images` folder. Supported formats:
- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.tiff`, `.tif`

The app will loop through all images in the folder continuously.

### 3. Run the Application

```bash
python app_windows.py
```

The app will:
- Process images from the `test_images` folder every 5 seconds
- Loop through all images continuously
- Start a web server at http://localhost:5000

### 4. Access the Web Interface

Open your browser and go to:
```
http://localhost:5000
```

## Configuration

You can modify these settings in `app_windows.py`:

- `CAPTURE_INTERVAL_SEC = 5` - Time between processing images (seconds)
- `TEST_IMAGES_FOLDER = Path("test_images")` - Folder containing test images
- `DEFAULT_TOLERANCE = 80` - Default similarity threshold (%)

## Differences from Raspberry Pi Version

- **No camera**: Reads images from folder instead
- **No picamera2**: Removed Raspberry Pi-specific dependency
- **Faster processing**: Default interval is 5 seconds (vs 60 seconds)
- **Fallback mode**: If paper detection fails, uses the full image for OCR

## Troubleshooting

### "No images found" error
- Make sure the `test_images` folder exists
- Add at least one image file to the folder
- Check that image files have supported extensions

### Tesseract OCR errors
- Make sure Tesseract is installed
- If needed, set the `tesseract_cmd` path in `app_windows.py`
- Check that Tesseract is in your system PATH

### Import errors
- Make sure all dependencies are installed: `pip install -r requirements_windows.txt`
- Try using a virtual environment

