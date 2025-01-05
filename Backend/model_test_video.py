#to test the working of the model on a video file

from ultralytics import YOLO
import cv2

#give the location of the model file
model = YOLO(r"D:\Mini-Project\TF LITE MODEL\Train 1 25 dec\best.pt")

video_path = r"D:\Mini-Project\videoplayback.mp4" #location of the video
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter to save output 
output_path = r"D:\Mini-Project\output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 model on the current frame
    results = model(frame)

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()

    # Save the annotated frame to output video
    out.write(annotated_frame)

    # Display the frame
    cv2.imshow("YOLOv8 Video Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()