FROM python:3.11-slim

# Install system dependencies including Tesseract OCR and OpenCV dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p static/uploads static/test_uploads static/test_processed

# Expose port (Render will set PORT env var)
EXPOSE 5000

# Run the application
CMD gunicorn app:app --bind 0.0.0.0:$PORT

