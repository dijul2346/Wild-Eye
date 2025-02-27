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
import gc

# Initialize YOLO model
model = YOLO(r"D:\Mini-Project\Modle\v12 26-2 2\weights\best.pt")  # Model path

# Initialize video capture
video_path = r"D:\Mini-Project\Modle\best.pt"  # 0 for webcam, or video path
cap = cv2.VideoCapture(2)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_FPS, fps)

# Parameters
fps = 30  # FPS
buffer_frames = fps * 2  # 2 seconds buffer
frame_buffer = deque(maxlen=buffer_frames)  # Circular buffer for frames
frame_skip = 1  # Frame skip rate

# Thread-safe global variables
detection_count = 0  # Total detections
last_detection_time = 0  # Last detection time
cooldown_period = 40  # Cooldown period in seconds
current_animal_detected = False  # Current detection status
last_detected_label = None  # Last detected label
detection_lock = threading.Lock()  # Lock for thread-safe access to global variables

# Initialize toast notifier
toast = ToastNotifier()

# Output directory
output_dir = r"D:\Mini-Project\Detections"  # Output directory for detections
os.makedirs(output_dir, exist_ok=True)

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(r"D:\Mini-Project\Firebase\wild-eye-f8551-firebase-adminsdk-fbsvc-8ff1c5d3b3.json")  # Firebase credentials path
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Configure Cloudinary
  
)

# Cloudinary upload function
def upload_to_cloudinary(file_path, resource_type="auto"):
    try:
        print(f"Uploading file: {file_path}")  # Debug print
        response = upload(file_path, resource_type=resource_type)
        print(f"Upload response: {response}")  # Debug print
        return response['secure_url']
    except Exception as e:
        print(f"Error uploading to Cloudinary: {e}")
        return None

# Firestore function
def save_detection_to_firestore(timestamp, label, image_url, video_url):
    try:
        detection_data = {
            "timestamp": timestamp,
            "label": label,
            "image_url": image_url,
            "video_url": video_url,
            "sent": False,
            "verified": False
        }
        db.collection("detections").add(detection_data)
        print("Detection saved to Firestore.")
    except Exception as e:
        print(f"Error saving to Firestore: {e}")

# Function to handle video upload in a separate thread
def upload_video_async(video_path, image_path, timestamp, label, conf):
    try:
        # Cloudinary upload
        image_url = upload_to_cloudinary(image_path, resource_type="image")
        video_url = upload_to_cloudinary(video_path, resource_type="video")
        print(f"Video upload response: {video_url}")  # Debug print

        # Firestore
        save_detection_to_firestore(timestamp, label, image_url, video_url)

        # Windows notification
        
    except Exception as e:
        print(f"Error during video upload: {e}")
    finally:
        # Clean up files after upload
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(image_path):
            os.remove(image_path)
        gc.collect()  # Force garbage collection

# Thread-safe queue for frames
frame_queue = queue.Queue(maxsize=10)  # Adjust maxsize based on memory constraints

# Function to perform YOLO inference in a separate thread
def inference_worker():
    global last_detection_time, last_detected_label

    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        # Perform YOLO inference
        results = model(frame)
        detections = results[0].boxes

        # Process detections
        for box in detections:
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class index
            label = model.names[cls]  # Class label
            xyxy = box.xyxy[0].tolist()  # Bounding box coordinates (x1, y1, x2, y2)

            # Confidence threshold
            if conf > 0.8:
                current_time = time.time()

                # Thread-safe access to global variables
                with detection_lock:
                    # Check if the same animal was detected within the cooldown period
                    if (current_time - last_detection_time) < cooldown_period and label == last_detected_label:
                        continue  # Skip this detection

                    # Update detection tracking
                    last_detection_time = current_time
                    last_detected_label = label

                # Draw bounding box and label on the frame
                x1, y1, x2, y2 = map(int, xyxy)  # Convert coordinates to integers
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Draw label and confidence

                # Save image
                timestamp = time.strftime("%d%m%Y_%H-%M-%S")  # Replace colons with hyphens
                image_path = os.path.join(output_dir, f"detected_{timestamp}.jpg")
                print(f"Saving image to: {image_path}")  # Debug print
                cv2.imwrite(image_path, frame)

                # Save video
                video_path = os.path.join(output_dir, f"detected_{timestamp}.mp4")
                print(f"Saving video to: {video_path}")  # Debug print
                out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
                toast.show_toast("Animal Detected", f"{label} detected with confidence {conf:.2f}.", duration=10)

                # Create a copy of the frame_buffer to avoid mutation during iteration
                buffered_frames = list(frame_buffer)  # Copy frames from the buffer
                for buffered_frame in buffered_frames:
                    # Draw bounding boxes on buffered frames
                    cv2.rectangle(buffered_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(buffered_frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    out.write(buffered_frame)
                out.release()

                # Start a new thread for video upload
                upload_thread = threading.Thread(
                    target=upload_video_async,
                    args=(video_path, image_path, timestamp, label, conf)
                )
                upload_thread.daemon = True  # Daemonize thread to avoid blocking program exit
                upload_thread.start()

        frame_queue.task_done()

# Start the inference worker thread
inference_thread = threading.Thread(target=inference_worker)
inference_thread.daemon = True
inference_thread.start()

# Main loop to capture and display frames
def detection_loop():
    global detection_count, current_animal_detected

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Add the current frame to the buffer
        frame_buffer.append(frame.copy())

        # Add the frame to the inference queue
        if not frame_queue.full():
            frame_queue.put(frame.copy())

        # Display the frame
        cv2.imshow("Live Animal Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()

# Start the detection loop
detection_loop()

# Cleanup
frame_queue.put(None)  # Signal the inference thread to exit
inference_thread.join()
