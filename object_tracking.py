import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------------
# Simple SORT Tracker (basic version)
# -------------------------------
class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = np.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x1, y1, x2, y2, id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count])
                self.id_count += 1

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids


# -------------------------------
# Load YOLO Model
# -------------------------------
model = YOLO("yolov8n.pt")  # lightweight model

# -------------------------------
# Start Video Capture (0 = webcam)
# -------------------------------
cap = cv2.VideoCapture(0)

tracker = Tracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------------------
    # Object Detection
    # -------------------------------
    results = model(frame)

    detections = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            detections.append([x1, y1, x2, y2])

    # -------------------------------
    # Tracking
    # -------------------------------
    tracked_objects = tracker.update(detections)

    # -------------------------------
    # Draw Results
    # -------------------------------
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Object Detection & Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()