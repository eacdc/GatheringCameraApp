# Deploying to Render

This guide explains how to deploy the Gathering Camera App to Render.

## Changes Made for Render Compatibility

1. **Removed Raspberry Pi dependencies**: Removed `picamera2` which requires physical camera hardware
2. **Added image upload API**: Users can now upload images via the web interface
3. **Updated requirements**: Added `gunicorn` for production server
4. **Environment variable support**: App uses `PORT` environment variable (set by Render)

## Deployment Steps

### 1. Push to GitHub

Make sure your code is pushed to a GitHub repository:

```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 2. Deploy on Render

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account if not already connected
4. Select your repository
5. Configure the service:
   - **Name**: `gathering-camera-app` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
6. Click **"Create Web Service"**

### 3. Install Tesseract OCR

Render uses Ubuntu-based containers. Add this to your build command to install Tesseract:

**Option A: In Render Dashboard**
- Go to your service settings
- Update **Build Command** to:
  ```
  apt-get update && apt-get install -y tesseract-ocr && pip install -r requirements.txt
  ```

**Option B: Using render.yaml** (already configured)
- The `render.yaml` file includes the build command
- Render will automatically use it if present

### 4. Environment Variables

No environment variables are required. The `PORT` variable is automatically set by Render.

## Usage

Once deployed:

1. Visit your Render service URL (e.g., `https://gathering-camera-app.onrender.com`)
2. Upload an image using the upload form
3. The app will:
   - Detect paper in the image
   - Run OCR to extract text
   - Compare with master text (if set)
   - Display results

## File Structure

```
.
├── app.py              # Main Flask application (Render-compatible)
├── requirements.txt    # Python dependencies
├── Procfile            # Process file for Render
├── render.yaml         # Render configuration
├── templates/
│   └── index.html      # Web interface with upload feature
├── static/
│   ├── uploads/        # Uploaded images (created automatically)
│   └── latest_paper.jpg # Latest processed image
└── master_text.txt     # Master text for comparison
```

## Differences from Raspberry Pi Version

- **No automatic capture**: Images must be uploaded manually
- **No camera hardware**: Works on any cloud platform
- **Same OCR and comparison logic**: All processing features remain the same

## Troubleshooting

### Tesseract not found
- Make sure the build command includes `apt-get install -y tesseract-ocr`
- Check Render logs for installation errors

### Upload fails
- Check file size limits (Render free tier has limits)
- Ensure `static/uploads/` directory exists (created automatically)

### Port errors
- Render automatically sets `PORT` environment variable
- The app uses `os.environ.get("PORT", 5000)` to handle this

## Notes

- Free tier services on Render spin down after 15 minutes of inactivity
- First request after spin-down may take longer to respond
- Consider upgrading to paid tier for always-on service

