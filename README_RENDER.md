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
5. Render will automatically detect the `Dockerfile` and `render.yaml`:
   - **Dockerfile**: Installs Tesseract OCR and Python dependencies
   - **render.yaml**: Configures the service settings
6. Click **"Create Web Service"** (or Render will auto-configure from `render.yaml`)

**Note**: The deployment uses Docker to install Tesseract OCR, which is required because Render's build environment doesn't allow `apt-get` commands directly.

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
├── Dockerfile          # Docker configuration with Tesseract OCR
├── Procfile            # Process file for Render (optional)
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
- The Dockerfile automatically installs Tesseract OCR
- If you see Tesseract errors, check that the Dockerfile is being used
- Verify in Render logs that the Docker build completed successfully

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

