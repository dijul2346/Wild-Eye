import sys
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
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QSlider, QListWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

# Initialize YOLO model
model = YOLO(r"D:\Mini-Project\Modle\train51\weights\best.pt")  

# Initialize video capture
video_path = r"D:\Mini-Project\Modle\best.pt"
cap = None  # Will be initialized when an input stream is selected
fps = 30
buffer_frames = fps * 5  # 5-second video buffer
frame_buffer = deque(maxlen=buffer_frames)
frame_skip = 1  # Frame skip rate

# Global variables
detection_count = 0
last_detection_time = 0
cooldown_period = 40
current_animal_detected = False
last_detected_label = None
detection_lock = threading.Lock()

toast = ToastNotifier()

# Output folder
output_dir = r"D:\Mini-Project\Detections"
os.makedirs(output_dir, exist_ok=True)

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(r"D:\Mini-Project\Firebase\wild-eye-f8551-firebase-adminsdk-fbsvc-8ff1c5d3b3.json")  # Firebase credentials path
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Configure Cloudinary
cloudinary.config(
    cloud_name="dondw5a6f", 
    api_key="321615463764633",        
    api_secret="6EnumTyZXO9vPoyMNxQDaonWJRo"  
)

dummy_locations = [
    {"name": "Muthanga", "link": "https://maps.app.goo.gl/YTzLAYZ7tAbLXkZc9"},
    {"name": "Aralam", "link": "https://maps.app.goo.gl/iJxLBb4YLyKx8aH88"},
    {"name": "Nelliyampathy", "link": "https://maps.app.goo.gl/ZywNpEe9qPfQqz1g7"},
    {"name": "Kakayam", "link": "https://maps.app.goo.gl/mUDd8sF1VRJ2YMPv5"},
    {"name": "Tholpetty", "link": "https://maps.app.goo.gl/pcT39qoyVwCuHY6X6"}
]

def get_random_location():
    return random.choice(dummy_locations)

def upload_to_cloudinary(file_path, resource_type="auto", transformation=None):
    try:
        print(f"Uploading file: {file_path}")  # Debug print
        response = upload(file_path, resource_type=resource_type, transformation=transformation)
        print(f"Upload response: {response}")  # Debug print
        return response['secure_url']
    except Exception as e:
        print(f"Error uploading to Cloudinary: {e}")
        return None

def save_detection_to_firestore(timestamp, label, image_url, video_url):
    try:
        location_data = get_random_location()
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
        db.collection("detections").add(detection_data)
        print("Detection saved to Firestore.")
    except Exception as e:
        print(f"Error saving to Firestore: {e}")

def upload_video_async(video_path, image_path, timestamp, label, conf):
    try:
        # Upload image
        image_url = upload_to_cloudinary(image_path, resource_type="image")

        # Upload video with transformation
        video_url = upload_to_cloudinary(
            video_path,
            resource_type="video",
            transformation=[
                {"width": 640, "height": 360, "crop": "scale"},
                {"format": "mp4"}
            ]
        )
        print(f"Video upload response: {video_url}")  # Debug print

        # Save detection to Firestore
        save_detection_to_firestore(timestamp, label, image_url, video_url)
    except Exception as e:
        print(f"Error during video upload: {e}")
    finally:
        # Clean up files
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(image_path):
            os.remove(image_path)
        gc.collect()  # Force garbage collection

frame_queue = queue.Queue(maxsize=10)

def inference_worker():
    global last_detection_time, last_detected_label

    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        results = model(frame)
        detections = results[0].boxes

        for box in detections:
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = model.names[cls]
            xyxy = box.xyxy[0].tolist()

            if conf > 0.8:  # Minimum confidence rate
                current_time = time.time()

                with detection_lock:
                    if (current_time - last_detection_time) < cooldown_period and label == last_detected_label:
                        continue

                    last_detection_time = current_time
                    last_detected_label = label

                x1, y1, x2, y2 = map(int, xyxy)

                # Save image with bounding box
                timestamp = time.strftime("%d-%m-%Y--%H-%M-%S")
                image_path = os.path.join(output_dir, f"detected_{timestamp}.jpg")
                print(f"Saving image to: {image_path}")
                image_with_box = frame.copy()
                cv2.rectangle(image_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_with_box, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imwrite(image_path, image_with_box)

                # Save video without bounding box
                video_path = os.path.join(output_dir, f"detected_{timestamp}.mp4")
                print(f"Saving video to: {video_path}")
                out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
                toast.show_toast("Animal Detected", f"{label} detected with confidence {conf:.2f}.", duration=10)

                buffered_frames = list(frame_buffer)
                for buffered_frame in buffered_frames:
                    out.write(buffered_frame)
                out.release()

                upload_thread = threading.Thread(
                    target=upload_video_async,
                    args=(video_path, image_path, timestamp, label, conf)
                )
                upload_thread.daemon = True
                upload_thread.start()

        frame_queue.task_done()

inference_thread = threading.Thread(target=inference_worker)
inference_thread.daemon = True
inference_thread.start()

def preprocess_frame(frame):
    frame = cv2.resize(frame, (640, 640))
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return frame

class AnimalDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Animal Detection")
        self.setGeometry(100, 100, 800, 600)

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layouts
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Video display
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)

        # Input stream selection
        self.input_stream_combo = QComboBox(self)
        self.input_stream_combo.addItems(["Webcam", "Phone", "OBS Studio", "RTSP Link"])
        self.input_stream_combo.currentIndexChanged.connect(self.change_input_stream)
        self.layout.addWidget(self.input_stream_combo)

        # RTSP links dropdown
        self.rtsp_combo = QComboBox(self)
        self.layout.addWidget(QLabel("Select RTSP Link:"))
        self.layout.addWidget(self.rtsp_combo)
        self.fetch_rtsp_links()

        # Recent detections list
        self.detections_list = QListWidget(self)
        self.layout.addWidget(QLabel("Recent Detections:"))
        self.layout.addWidget(self.detections_list)
        self.fetch_recent_detections()

        # Cooldown period slider
        self.cooldown_slider = QSlider(Qt.Horizontal, self)
        self.cooldown_slider.setMinimum(10)
        self.cooldown_slider.setMaximum(120)
        self.cooldown_slider.setValue(cooldown_period)
        self.cooldown_slider.valueChanged.connect(self.update_cooldown_period)
        self.layout.addWidget(QLabel("Cooldown Period (seconds):"))
        self.layout.addWidget(self.cooldown_slider)

        # Remaining cooldown time display
        self.cooldown_label = QLabel(f"Remaining Cooldown: {cooldown_period} seconds", self)
        self.layout.addWidget(self.cooldown_label)

        # Timer for updating the video feed and cooldown
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def fetch_rtsp_links(self):
        try:
            # Fetch RTSP links from the "addCCTV" collection
            rtsp_links = db.collection("addCCTV").stream()
            for link in rtsp_links:
                link_data = link.to_dict()
                rtsp_link = link_data.get("rtspLink")
                if rtsp_link:
                    self.rtsp_combo.addItem(rtsp_link)
        except Exception as e:
            print(f"Error fetching RTSP links: {e}")

    def fetch_recent_detections(self):
        try:
            # Fetch recent detections from the "detections" collection
            detections = db.collection("detections").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(10).stream()
            for detection in detections:
                detection_data = detection.to_dict()
                label = detection_data.get("label")
                timestamp = detection_data.get("timestamp")
                if label and timestamp:
                    self.detections_list.addItem(f"{label} - {timestamp}")
        except Exception as e:
            print(f"Error fetching recent detections: {e}")

    def change_input_stream(self, index):
        global cap
        if cap is not None and cap.isOpened():
            cap.release()

        if index == 0:  # Webcam
            cap = cv2.VideoCapture(0)
        elif index == 1:  # Phone
            cap = cv2.VideoCapture(1)
        elif index == 2:  # OBS Studio
            cap = cv2.VideoCapture(2)
        elif index == 3:  # RTSP Link
            selected_rtsp = self.rtsp_combo.currentText()
            if selected_rtsp:
                cap = cv2.VideoCapture(selected_rtsp)

        if cap is not None and cap.isOpened():
            self.timer.start(30)  # Start the timer to update frames

    def update_cooldown_period(self, value):
        global cooldown_period
        cooldown_period = value
        self.cooldown_label.setText(f"Remaining Cooldown: {cooldown_period} seconds")

    def update_frame(self):
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                processed_frame = preprocess_frame(frame)
                frame_buffer.append(processed_frame.copy())
                if not frame_queue.full():
                    frame_queue.put(processed_frame.copy())

                # Convert the frame to QImage
                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(convert_to_Qt_format))

                # Update cooldown label
                remaining_cooldown = max(0, cooldown_period - (time.time() - last_detection_time))
                self.cooldown_label.setText(f"Remaining Cooldown: {int(remaining_cooldown)} seconds")

    def closeEvent(self, event):
        self.timer.stop()
        if cap is not None and cap.isOpened():
            cap.release()
        frame_queue.put(None)
        inference_thread.join()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnimalDetectionApp()
    window.show()
    sys.exit(app.exec_())
