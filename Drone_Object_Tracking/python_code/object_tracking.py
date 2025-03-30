import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("./datasets/drone_footage.mp4")
frame_width = int(cap.get(3))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2

            # Define zones
            left_zone = frame_width // 3
            right_zone = (frame_width // 3) * 2

            # Determine movement
            if center_x < left_zone:
                direction = "Turn Left"
            elif center_x > right_zone:
                direction = "Turn Right"
            else:
                direction = "Stay Centered"

            # Display movement command
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2)

    cv2.imshow("Drone Object Tracking with Control", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
