# utils.py

import cv2
import os
import time

def save_frame(frame, base_path="captured_frames"):
    """
    將偵測到的畫面保存到指定路徑。

    Args:
        frame (numpy.ndarray): 要保存的影像幀。
        base_path (str): 畫面將被保存的目錄。
                         預設為 "captured_frames"。
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

    Args:
        results (list): YOLOv8 偵測結果物件的列表。
                        預期至少包含 results[0] 及其 .boxes 和 .names 屬性。

    Returns:
        dict: 字典，其中鍵為類別名稱，值為其計數。
    """
    counts = {}
    # 檢查結果是否存在且是否有偵測到的邊界框
    if results and results[0].boxes:
        for r in results[0].boxes:
            # 獲取類別 ID 並查找其名稱
            cls = int(r.cls[0])
            name = results[0].names[cls]
            # 增加此類別的計數
            counts[name] = counts.get(name, 0) + 1
    return counts