import cv2
import numpy as np
from ultralytics import YOLO
import cvzone



# Load the YOLO11 model
model = YOLO(|"best.pt")
names=model.names
# Open the video file (use video file or webcam, here using webcam)
cap = cv2.VideoCapture('vid.mp4')
count=0


while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

#    count += 1
#    if count % 2 != 0:
#        continue

    frame = cv2.resize(frame, (1020, 500))
    
    # Run YOLO11 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True)

    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the boxes (x, y, w, h), class IDs, track IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score
       
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = names[class_id]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cvzone.putTextRect(frame,f'{track_id}',(x1,y2),1,1)
            cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
       break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
