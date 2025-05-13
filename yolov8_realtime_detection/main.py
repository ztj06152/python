from ultralytics import YOLO
import cv2
from utils import save_frame, count_objects
import time
import csv
import os
import threading
import matplotlib.pyplot as plt

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# 計時
prev_infer_time = 0
prev_plot_time = 0

# 統計
csv_file = 'results.csv'
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Class', 'Count'])

total_counts = {}
last_results = None
latest_plot_data = {}

# ================= 圖表執行緒 ================= #
def plot_thread():
    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("物件偵測統計圖")
    while True:
        ax.clear()
        classes = list(latest_plot_data.keys())
        values = list(latest_plot_data.values())
        ax.bar(classes, values, color='skyblue')
        ax.set_title("Object Detection Counts")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.pause(0.1)
        time.sleep(0.1)  # 減少 CPU 使用

# 啟動圖表執行緒
threading.Thread(target=plot_thread, daemon=True).start()
# ================================================= #

while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()

    # 每 1 秒推論一次
    if curr_time - prev_infer_time >= 1:
        last_results = model(frame)
        counts = count_objects(last_results)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        for label, count in counts.items():
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, label, count])
            total_counts[label] = total_counts.get(label, 0) + count

        prev_infer_time = curr_time

    # 每 3 秒更新圖表統計資料
    if curr_time - prev_plot_time >= 3:
        latest_plot_data = total_counts.copy()
        prev_plot_time = curr_time

    # 顯示影像畫面
    if last_results:
        annotated_frame = last_results[0].plot()
        annotated_frame = cv2.resize(annotated_frame, (800, 600))
        cv2.imshow("YOLOv8 Detection", annotated_frame)

    # 鍵盤控制
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('s') and last_results:
        save_frame(annotated_frame)

cap.release()
cv2.destroyAllWindows()
