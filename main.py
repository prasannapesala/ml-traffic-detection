import cv2
from ultralytics import YOLO
import joblib

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

# Load ML model
ml_model = joblib.load("model.pkl")

# Load video
cap = cv2.VideoCapture("traffic.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)

    # Frame size
    h, w, _ = frame.shape

    # Lane counts
    l1 = l2 = l3 = l4 = 0

    # Detection loop
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = yolo_model.names[cls]

            if label in ["car", "truck", "bus", "motorbike"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # Center of vehicle
                center_x = (x1 + x2) // 2

                # Assign lane
                if center_x < w/4:
                    l1 += 1
                elif center_x < w/2:
                    l2 += 1
                elif center_x < 3*w/4:
                    l3 += 1
                else:
                    l4 += 1

    # Total vehicles
    vehicle_count = l1 + l2 + l3 + l4

    # ML prediction
    green_time = ml_model.predict([[l1, l2, l3, l4]])[0]

    # Decide which lane gets green
    lanes = [l1, l2, l3, l4]
    max_value = max(lanes)
    max_lane_index = lanes.index(max_value)
    green_lane = max_lane_index + 1

    # Display text
    cv2.putText(frame, f"Total Vehicles: {vehicle_count}", (20,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    cv2.putText(frame, f"L1:{l1} L2:{l2} L3:{l3} L4:{l4}", (20,70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.putText(frame, f"Green Time: {green_time:.2f}s", (20,110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.putText(frame, f"Green Lane: L{green_lane}", (20,150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    # Draw lane divisions
    cv2.line(frame, (w//4, 0), (w//4, h), (255,255,255), 1)
    cv2.line(frame, (w//2, 0), (w//2, h), (255,255,255), 1)
    cv2.line(frame, (3*w//4, 0), (3*w//4, h), (255,255,255), 1)

    # Highlight selected lane
    lane_width = w // 4

    if green_lane == 1:
        cv2.rectangle(frame, (0,0), (lane_width,h), (0,255,0), 4)
    elif green_lane == 2:
        cv2.rectangle(frame, (lane_width,0), (2*lane_width,h), (0,255,0), 4)
    elif green_lane == 3:
        cv2.rectangle(frame, (2*lane_width,0), (3*lane_width,h), (0,255,0), 4)
    elif green_lane == 4:
        cv2.rectangle(frame, (3*lane_width,0), (w,h), (0,255,0), 4)

    # Show frame
    cv2.imshow("Smart Traffic System", frame)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
