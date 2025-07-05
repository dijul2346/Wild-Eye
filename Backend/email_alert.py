import cv2
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from ultralytics import YOLO
from dotenv import load_dotenv  # Added for .env support

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Function to send an email with an attachment
def send_email_with_attachment(subject, body, recipient_email, attachment_path):
    sender_email = os.getenv('SENDER_EMAIL')
    sender_password = os.getenv('SENDER_PASSWORD')
    # recipient_email is now passed as argument, but can be loaded from env if needed
    # recipient_email = os.getenv('RECIPIENT_EMAIL')

    # Create the email
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient_email
    message['Subject'] = subject

    # Add the body text
    message.attach(MIMEText(body, 'plain'))

    # Attach the file
    if os.path.exists(attachment_path):
        with open(attachment_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename={os.path.basename(attachment_path)}'
        )
        message.attach(part)

    # Connect to the SMTP server and send the email
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
            print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")


# Initialize YOLOv8 model
model = YOLO(r"D:\Mini-Project\TF LITE MODEL\Train 1 25 dec\best.pt")  # Replace with your YOLOv8 model path

# Set up video capture
#rtsp_url="rtsp://admin:admin@192.168.129.18:1935"
#cap = cv2.VideoCapture(rtsp_url)  # Use 0 for default webcam or provide video path
cap = cv2.VideoCapture(0)

# Change to .mpeg format for video output
out = cv2.VideoWriter("detected_animal.mkv", cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))  # 'MPG4' codec for .mpeg format

video_saved = False  # Flag to avoid multiple email sends
detection_count = 0  # Counter to track the number of detections

# Use environment variable for recipient email
default_recipient_email = os.getenv('RECIPIENT_EMAIL')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)
    detections = results[0].boxes

    for box in detections:
        conf = box.conf[0].item()  # Confidence score
        cls = int(box.cls[0].item())  # Class index
        label = model.names[cls]  # Get class label

        # Only process detections with confidence > 90%
        if conf > 0.85:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Save annotated frame to video
            out.write(frame)

            # Increment the detection count
            detection_count += 1

    # Display the annotated frame
    cv2.imshow("YOLOv8 Real-Time Detection", frame)

    # Check if the object has been detected more than 30 times
    if detection_count >= 30 and not video_saved:
        local_file_path = "detected_animal.mkv"  # Update file extension to .mpeg
        recipient_email = default_recipient_email  # Use env variable

        # Send email with video attachment
        send_email_with_attachment(
            subject="Animal Detected!",
            body="An animal has been detected. Check the attached video.",
            recipient_email=recipient_email,
            attachment_path=local_file_path
        )

        video_saved = True  # Set flag to prevent multiple emails

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup resources
cap.release()
out.release()
cv2.destroyAllWindows()
