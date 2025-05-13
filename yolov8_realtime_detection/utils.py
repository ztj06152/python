import cv2
import os
from datetime import datetime

def save_frame(frame, folder="snapshots"):
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"screenshot_{timestamp}.jpg")
    cv2.imwrite(path, frame)
    print(f"[INFO] Saved snapshot to {path}")

def count_objects(results):
    names = results[0].names
    counts = {}
    for box in results[0].boxes:
        cls_id = int(box.cls)
        cls_name = names[cls_id]
        counts[cls_name] = counts.get(cls_name, 0) + 1
    return counts
