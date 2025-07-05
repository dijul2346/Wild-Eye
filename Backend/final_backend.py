import cv2
import os
import threading
from collections import deque
from win10toast import ToastNotifier
import time
from ultralytics import YOLO
import cloudinary
from cloudinary.uploader import upload
import firebase_admin
from firebase_admin import credentials, firestore
import queue
import random
import gc
import numpy as np
from dotenv import load_dotenv  # Added for .env support

class AnimalDetectionBackend:
    def __init__(self):
        # Load environment variables
        load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
        # Use relative paths from environment variables or fallback
        model_path = os.getenv('MODEL_PATH', os.path.join('..', 'Model', 'best.pt'))
        self.model = YOLO(model_path)
        
        # Video configuration
        self.cap = None
        self.fps = 30
        self.buffer_frames = self.fps * 5  # 5 second buffer
        self.frame_buffer = deque(maxlen=self.buffer_frames)
        self.frame_skip = 1
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Detection tracking
        self.detection_count = 0
        self.last_detection_time = 0
        self.cooldown_period = 40
        self.current_animal_detected = False
        self.last_detected_label = None
        self.detection_lock = threading.Lock()
        
        # Notifications
        self.toast = ToastNotifier()
        
        # Output directory (relative)
        self.output_dir = os.getenv('OUTPUT_DIR', os.path.join('..', 'Detections'))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize cloud services
        self._initialize_cloud_services()
        
        # Thread management
        self.inference_thread = None
        self.capture_thread = None
        self.running = False
        
    def _initialize_cloud_services(self):
        """Initialize Firebase and Cloudinary connections"""
        firebase_credentials_path = os.getenv('FIREBASE_CREDENTIALS_PATH', os.path.join('..', '..', 'Firebase', 'wild-eye-f8551-firebase-adminsdk-fbsvc-8ff1c5d3b3.json'))
        if not firebase_admin._apps:
            cred = credentials.Certificate(firebase_credentials_path)
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        
        cloudinary.config(
            cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
            api_key=os.getenv('CLOUDINARY_API_KEY'),
            api_secret=os.getenv('CLOUDINARY_API_SECRET')
        )
    
    def get_random_location(self):
        """Return a random location from predefined set"""
        locations = [
            {"name": "Muthanga", "link": "https://maps.app.goo.gl/YTzLAYZ7tAbLXkZc9"},
            {"name": "Aralam", "link": "https://maps.app.goo.gl/iJxLBb4YLyKx8aH88"},
            {"name": "Nelliyampathy", "link": "https://maps.app.goo.gl/ZywNpEe9qPfQqz1g7"},
            {"name": "Kakayam", "link": "https://maps.app.goo.gl/mUDd8sF1VRJ2YMPv5"},
            {"name": "Tholpetty", "link": "https://maps.app.goo.gl/pcT39qoyVwCuHY6X6"}
        ]
        return random.choice(locations)
    
    def upload_to_cloudinary(self, file_path, resource_type="auto", transformation=None):
        """Upload a file to Cloudinary"""
        try:
            response = upload(file_path, resource_type=resource_type, transformation=transformation)
            return response['secure_url']
        except Exception as e:
            print(f"Error uploading to Cloudinary: {e}")
            return None
    
    def save_detection_to_firestore(self, timestamp, label, image_url, video_url):
        """Save detection metadata to Firestore"""
        try:
            location_data = self.get_random_location()
            detection_data = {
                "adminReply": False,
                "timestamp": timestamp,
                "label": label,
                "image_url": image_url,
                "video_url": video_url,
                "sent": False,
                "verified": False,
                "location": location_data["name"],
                "location_link": location_data["link"],
                "userViewed": False
            }
            self.db.collection("detections").add(detection_data)
            print("Detection saved to Firestore.")
        except Exception as e:
            print(f"Error saving to Firestore: {e}")
    
    def upload_video_async(self, video_path, image_path, timestamp, label, conf):
        """Handle async upload of detection media"""
        try:
            image_url = self.upload_to_cloudinary(image_path, resource_type="image")
            video_url = self.upload_to_cloudinary(
                video_path,
                resource_type="video",
                transformation=[
                    {"width": 640, "height": 360, "crop": "scale"},
                    {"format": "mp4"}
                ]
            )
            self.save_detection_to_firestore(timestamp, label, image_url, video_url)
        except Exception as e:
            print(f"Error during video upload: {e}")
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(image_path):
                os.remove(image_path)
            gc.collect()
    
    def inference_worker(self):
        """Process frames for object detection"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                if frame is None:
                    continue
                
                results = self.model(frame)
                detections = results[0].boxes
                
                for box in detections:
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = self.model.names[cls]
                    xyxy = box.xyxy[0].tolist()
                    
                    if conf > 0.8:  # Confidence threshold
                        current_time = time.time()
                        
                        with self.detection_lock:
                            if (current_time - self.last_detection_time) < self.cooldown_period and label == self.last_detected_label:
                                continue
                            
                            self.last_detection_time = current_time
                            self.last_detected_label = label
                        
                        # Save detection
                        self._save_detection(frame, xyxy, label, conf)
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in inference worker: {e}")
                continue
    
    def _save_detection(self, frame, xyxy, label, conf):
        """Handle saving a detection (image + video)"""
        x1, y1, x2, y2 = map(int, xyxy)
        timestamp = time.strftime("%d-%m-%Y--%H-%M-%S")
        
        # Save image
        image_path = os.path.join(self.output_dir, f"detected_{timestamp}.jpg")
        image_with_box = frame.copy()
        cv2.rectangle(image_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_with_box, f"{label} {conf:.2f}", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imwrite(image_path, image_with_box)
        
        # Save video
        video_path = os.path.join(self.output_dir, f"detected_{timestamp}.mp4")
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                             self.fps, (frame.shape[1], frame.shape[0]))
        
        self.toast.show_toast("Animal Detected", 
                            f"{label} detected with confidence {conf:.2f}.", 
                            duration=10)
        
        # Write buffered frames
        for buffered_frame in list(self.frame_buffer):
            out.write(buffered_frame)
        out.release()
        
        # Start upload thread
        upload_thread = threading.Thread(
            target=self.upload_video_async,
            args=(video_path, image_path, timestamp, label, conf)
        )
        upload_thread.daemon = True
        upload_thread.start()
    
    def preprocess_frame(self, frame):
        """Preprocess a video frame"""
        frame = cv2.resize(frame, (640, 640))
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        return frame
    
    def video_capture_worker(self, source):
        """Handle video capture in a dedicated thread"""
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        reconnect_delay = 5  # seconds
        
        while self.running and reconnect_attempts < max_reconnect_attempts:
            try:
                if self.cap is not None:
                    self.cap.release()
                
                # Special handling for RTSP streams
                if isinstance(source, str) and source.startswith('rtsp'):
                    self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                    # Set buffer size to minimize latency
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                else:
                    self.cap = cv2.VideoCapture(source)
                
                if not self.cap.isOpened():
                    raise RuntimeError(f"Failed to open video source: {source}")
                
                print(f"Successfully connected to video source: {source}")
                reconnect_attempts = 0
                
                while self.running:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Frame read failed, attempting to reconnect...")
                        break
                    
                    processed_frame = self.preprocess_frame(frame)
                    self.frame_buffer.append(processed_frame.copy())
                    
                    if not self.frame_queue.full():
                        try:
                            self.frame_queue.put_nowait(processed_frame.copy())
                        except queue.Full:
                            pass
                    
                    time.sleep(1/self.fps)  # Control frame rate
                
            except Exception as e:
                print(f"Video capture error: {e}")
                reconnect_attempts += 1
                if reconnect_attempts < max_reconnect_attempts:
                    print(f"Attempting to reconnect in {reconnect_delay} seconds...")
                    time.sleep(reconnect_delay)
        
        if reconnect_attempts >= max_reconnect_attempts:
            print("Max reconnection attempts reached. Video capture stopped.")
    
    def start(self, source=0):
        """Start the detection system"""
        self.running = True
        
        # Start inference thread
        self.inference_thread = threading.Thread(target=self.inference_worker)
        self.inference_thread.daemon = True
        self.inference_thread.start()
        
        # Start video capture thread
        self.capture_thread = threading.Thread(target=self.video_capture_worker, args=(source,))
        self.capture_thread.daemon = True
        self.capture_thread.start()
    
    def stop(self):
        """Stop the detection system"""
        self.running = False
        
        # Signal threads to stop
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        
        if self.inference_thread and self.inference_thread.is_alive():
            self.frame_queue.put(None)  # Signal to stop
            self.inference_thread.join(timeout=2)
        
        # Release video capture
        if self.cap is not None:
            self.cap.release()
        
        print("Detection system stopped.")
    
    def change_video_source(self, source):
        """Change the video source dynamically"""
        self.stop()
        self.start(source)
    
    def get_frame(self):
        """Get the latest frame for display"""
        if len(self.frame_buffer) > 0:
            return self.frame_buffer[-1]
        return None
    
    def fetch_rtsp_links(self):
        """Fetch available RTSP links from Firestore"""
        try:
            rtsp_links = []
            links_ref = self.db.collection("addCCTV").stream()
            for link in links_ref:
                link_data = link.to_dict()
                rtsp_link = link_data.get("rtspLink")
                if rtsp_link:
                    rtsp_links.append(rtsp_link)
            return rtsp_links
        except Exception as e:
            print(f"Error fetching RTSP links: {e}")
            return []
    
    def fetch_recent_detections(self, limit=10):
        """Fetch recent detections from Firestore"""
        try:
            detections = []
            detections_ref = self.db.collection("detections").order_by(
                "timestamp", direction=firestore.Query.DESCENDING).limit(limit).stream()
            for detection in detections_ref:
                detection_data = detection.to_dict()
                detections.append(detection_data)
            return detections
        except Exception as e:
            print(f"Error fetching recent detections: {e}")
            return []
