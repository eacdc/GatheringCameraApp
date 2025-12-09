# Desktop GUI Application - Quick Start

A fast, native desktop application for testing image OCR functionality on Windows.

## Installation

1. **Install Tesseract OCR** (if not already installed):
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Or use Chocolatey: `choco install tesseract`

2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements_windows.txt
   ```

## Running the Desktop App

Simply run:
```bash
python app_desktop.py
```

The desktop window will open immediately - no web server needed!

## Features

- **Fast & Responsive**: Native desktop UI, no browser overhead
- **Image Selection**: Click "Select Image" to choose an image file
- **OCR Processing**: Click "Process OCR" to extract text from the image
- **Text Comparison**: Compare OCR results with master text
- **Master Text Management**: Save and load reference text
- **Visual Feedback**: Color-coded status indicators and similarity scores

## Usage

1. **Select Image**: Click "📁 Select Image" button
2. **Process OCR**: Click "🔄 Process OCR" to extract text
3. **Set Master Text**: 
   - Type or paste reference text in the "Master Text" area
   - Click "💾 Save Master Text" to save it
   - Or click "📋 Load from OCR" to use current OCR result as master
4. **Compare**: Click "🔍 Compare Texts" to see similarity score
5. **Clear**: Click "🗑️ Clear" to reset and start over

## Advantages over Web Version

- ✅ **Faster**: No network overhead, direct processing
- ✅ **Offline**: Works completely offline
- ✅ **Native Feel**: Windows-native interface
- ✅ **No Server**: No need to start Flask server
- ✅ **Better Performance**: Direct file access, no HTTP overhead

## System Requirements

- Windows 10/11
- Python 3.7+
- Tesseract OCR installed
- All dependencies from `requirements_windows.txt`

