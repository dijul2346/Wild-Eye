# Wildlife Alert System

A real-time animal detection and alert system using YOLO, Firestore, and Cloudinary. This project captures video, detects wildlife, saves annotated images and videos, and uploads detection data to the cloud.

## Features

- Real-time animal detection using YOLO
- Video and image capture with bounding boxes
- Uploads media to Cloudinary
- Stores detection metadata in Firestore
- Windows toast notifications for detections
- Fetches RTSP camera links and recent detections from Firestore
- Configurable via `.env` file

## Project Structure

```
Backend/
    final_backend.py      # Main backend class (YOLO, Firestore, Cloudinary)
    backend.py            # (Optional) Script-based backend
    email_alert.py        # Email alert script
    .env                  # Environment variables (not tracked by git)
Model/
    best.pt               # YOLO model weights
App/
    wildlife_detection/   # Flutter app (optional)
```

## Setup

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   (You may need: `opencv-python`, `ultralytics`, `firebase-admin`, `cloudinary`, `python-dotenv`, `win10toast`, `numpy`)

3. **Configure environment variables**

   Create a `Backend/.env` file:
   ```
   MODEL_PATH=../Model/best.pt
   OUTPUT_DIR=../Detections
   FIREBASE_CREDENTIALS_PATH=../../Firebase/wild-eye-f8551-firebase-adminsdk-fbsvc-8ff1c5d3b3.json
   CLOUDINARY_CLOUD_NAME=your_cloud_name
   CLOUDINARY_API_KEY=your_api_key
   CLOUDINARY_API_SECRET=your_api_secret
   ```

4. **Run the backend**
   ```bash
   python Backend/final_backend.py
   ```

## Notes

- The `.env` file is ignored by git for security.
- All paths are relative for cross-platform and GitHub compatibility.
- For email alerts, see `email_alert.py` and add email credentials to `.env`.

## License

MIT License
