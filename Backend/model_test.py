#To test the live detection capabilities of the model

from ultralytics import YOLO
import cv2

#give the location of the model file
model = YOLO(r"D:\Mini-Project\TF LITE MODEL\Train 1 25 dec\best.pt")
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam or provide a video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 model on the current frame
    results = model(frame)

    # Filter detections with confidence > 90%
    for result in results:
        for box in result.boxes:
            if box.conf > 0.85:  # Confidence threshold
                # Draw bounding boxes and labels manually
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf.item()  # Confidence score
                cls = int(box.cls[0])   # Class index
                label = f"{model.names[cls]}: {conf:.2f}"  # Label with class name and confidence

                # Draw the box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLOv8 Real-Time Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()