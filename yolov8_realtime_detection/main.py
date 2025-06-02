import matplotlib
try:
    matplotlib.use('TkAgg')  # 通常適用於大多數桌面環境
except ImportError:
    print("Tkinter 後端不可用，嘗試其他後端或確保已安裝 tkinter。")
    matplotlib.use('Agg')  # 如果 Tkinter 不可用，切換到非互動模式

import matplotlib.pyplot as plt
plt.rc("font", family="Microsoft JhengHei")
from ultralytics import YOLO
import cv2
import time
import csv
import os
import threading
import queue  # 引入 queue 模組用於執行緒間通訊


# --- BEGIN: 假設這是您的 utils.py 中的函數，為了程式完整性直接包含在此 ---
def save_frame(frame, base_path="captured_frames"):
    """
    將偵測到的畫面保存到指定路徑。
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(base_path, f"frame_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"畫面已保存至 {filename}")


def count_objects(results):
    """
    從 YOLOv8 推論結果中計算各類別物件的數量。
    """
    counts = {}
    if results and results[0].boxes:
        for r in results[0].boxes:
            cls = int(r.cls[0])
            name = results[0].names[cls]
            counts[name] = counts.get(name, 0) + 1
    return counts


# --- END: utils.py 函數 ---

# 加載 YOLOv8 模型
model = YOLO("yolov8n.pt")
# 開啟攝像頭
cap = cv2.VideoCapture(0)

# 檢查攝像頭是否成功開啟
if not cap.isOpened():
    print("錯誤：無法開啟攝像頭，請檢查設備連接或權限。")
    exit()

# 計時變數
prev_infer_time = 0  # 上次推論時間
prev_plot_time = 0  # 上次圖表更新時間

# 統計數據設置
csv_file = 'results.csv'
# 如果 CSV 文件不存在，則創建並寫入表頭
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Class', 'Count'])

# 全局變數用於累積總計數和圖表數據
total_counts = {}
latest_plot_data = {}
last_annotated_frame = None  # 用於儲存帶有偵測結果的最新畫面，供主迴圈顯示

# ================= 執行緒間通訊佇列與事件 ================= #
# 畫面佇列：主執行緒將最新畫面傳遞給推論執行緒
frame_queue = queue.Queue(maxsize=1)
# 結果佇列：推論執行緒將偵測結果傳遞回主執行緒
result_queue = queue.Queue(maxsize=1)
# 停止事件：用於通知所有執行緒安全地終止
stop_event = threading.Event()


# ========================================================= #

# ================= 推論執行緒函數 ================= #
def inference_thread_func():
    """
    獨立執行緒，負責從畫面佇列獲取畫面並執行 YOLOv8 推論。
    推論頻率為每 1.5 秒一次。
    """
    global prev_infer_time  # 宣告使用全局變數

    print("推論執行緒已啟動。")
    while not stop_event.is_set():  # 當停止事件未被設定時，持續運行
        try:
            # 從畫面佇列獲取最新畫面。設置 timeout 以便定期檢查 stop_event。
            # 如果佇列為空，會在 timeout 後拋出 Empty 異常。
            frame = frame_queue.get(timeout=0.05)  # 稍微縮短 timeout 以提高響應性
            curr_infer_time = time.time()

            # 每 1.5 秒進行一次物件推論 (核心變更點)
            if curr_infer_time - prev_infer_time >= 1.5:
                # 執行 YOLOv8 模型推論
                results = model(frame)
                # 計算物件數量
                counts = count_objects(results)
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

                # 繪製偵測框和標籤到畫面
                annotated_frame_for_display = results[0].plot()

                # 將偵測結果（帶框畫面、計數、時間戳）放入結果佇列
                if not result_queue.full():
                    try:
                        result_queue.put_nowait((annotated_frame_for_display, counts, timestamp))
                    except queue.Full:
                        pass  # 如果佇列滿了，表示主執行緒還沒處理完，暫時跳過

                prev_infer_time = curr_infer_time

        except queue.Empty:
            # 佇列為空，繼續循環等待新畫面
            continue
        except Exception as e:
            # 捕獲推論過程中的任何錯誤
            print(f"推論執行緒錯誤: {e}")
            break  # 發生錯誤時終止執行緒
    print("推論執行緒已停止。")


# 啟動推論執行緒 (設置為 Daemon 模式，主程式結束時會自動終止)
threading.Thread(target=inference_thread_func, daemon=True).start()


# ========================================================= #

# ================= 圖表更新執行緒函數 ================= #
def plot_thread_func():
    """
    獨立執行緒，負責實時更新 Matplotlib 統計圖表。
    圖表刷新頻率為每 2.5 秒一次。
    """
    plt.ion()  # 開啟互動模式，以便圖表實時更新
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("物件偵測統計圖")

    print("圖表執行緒已啟動。")
    while not stop_event.is_set():  # 當停止事件未被設定時，持續運行
        ax.clear()  # 清除上一次繪製的圖表
        if latest_plot_data:  # 確保有數據才繪製
            classes = list(latest_plot_data.keys())
            values = list(latest_plot_data.values())
            ax.bar(classes, values, color='skyblue')
        else:
            # 如果暫無數據，顯示提示
            ax.text(0.5, 0.5, "等待偵測數據...",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)

        ax.set_title("物件偵測計數")
        ax.set_xlabel("類別")
        ax.set_ylabel("計數")
        plt.tight_layout()  # 自動調整佈局以防止標籤重疊

        # 這裡的 pause 時間可以短，因為圖表數據的更新頻率由主迴圈控制 (2.5秒)
        # plt.pause(0.1) 讓圖表視窗有機會處理事件並重繪
        plt.pause(0.1)

    plt.close('all')  # 關閉所有 Matplotlib 視窗
    print("圖表執行緒已停止。")


# 啟動圖表執行緒 (設置為 Daemon 模式，主程式結束時會自動終止)
threading.Thread(target=plot_thread_func, daemon=True).start()
# ========================================================= #

# ================= 主程式迴圈 ================= #
print("主程式迴圈啟動...")
while True:
    ret, frame = cap.read()  # 持續從攝像頭讀取畫面
    if not ret:
        print("無法從攝像頭讀取畫面，可能是流結束或設備斷開。")
        break

    curr_time = time.time()

    # 將最新畫面放入 `frame_queue` 供推論執行緒使用。
    # `put_nowait()` 是非阻塞的。如果佇列滿了，則會跳過這個畫面，避免阻塞主迴圈。
    if not frame_queue.full():
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # 如果佇列滿了，就丟棄這個畫面，等待下一個

    # 嘗試從 `result_queue` 獲取最新偵測結果。
    # `get_nowait()` 是非阻塞的。如果佇列為空，會立即拋出 `Empty` 異常。
    try:
        annotated_frame_from_infer, counts_from_infer, timestamp_from_infer = result_queue.get_nowait()
        # 儲存帶有偵測結果的畫面，用於顯示
        last_annotated_frame = cv2.resize(annotated_frame_from_infer, (800, 600))

        # 將偵測結果寫入 CSV (在主執行緒處理，避免檔案鎖定問題)
        for label, count in counts_from_infer.items():
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp_from_infer, label, count])
            total_counts[label] = total_counts.get(label, 0) + count

    except queue.Empty:
        # 如果結果佇列為空，表示推論執行緒還沒有新的結果傳回，繼續顯示舊畫面
        pass

        # 每 2.5 秒更新圖表數據 (核心變更點)
    if curr_time - prev_plot_time >= 2.5:
        latest_plot_data = total_counts.copy()  # 複製一份，避免執行緒間直接修改數據
        print(f"圖表數據更新，當前總計數: {latest_plot_data}")  # 添加打印以確認數據更新
        prev_plot_time = curr_time

    # 顯示影像畫面
    # 如果有偵測結果畫面，顯示帶框的畫面；否則顯示原始畫面
    if last_annotated_frame is not None:
        cv2.imshow("YOLOv8 物件偵測", last_annotated_frame)
    else:
        # 在還沒有任何偵測結果時，顯示原始且縮放過的畫面
        resized_frame = cv2.resize(frame, (800, 600))
        cv2.imshow("YOLOv8 物件偵測", resized_frame)

    # 鍵盤控制
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 按 'q' 或 'Esc' 鍵退出
        break
    elif key == ord('s') and last_annotated_frame is not None:  # 按 's' 鍵保存帶框畫面
        save_frame(last_annotated_frame)

# ================= 程式結束，清理資源 ================= #
print("正在停止所有執行緒並釋放資源...")
stop_event.set()  # 發送停止信號給所有執行緒
time.sleep(0.5)  # 給執行緒一點時間來停止
cap.release()  # 釋放攝像頭資源
cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗
plt.close('all')  # 關閉所有 Matplotlib 圖表視窗
print("程式已成功結束。")