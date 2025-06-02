# Real-time Object Detection and Analysis System

This project implements a real-time object detection and analysis system using YOLOv8, OpenCV, and Matplotlib. It continuously monitors a webcam feed, detects objects, logs detection counts to a CSV file, and visualizes the statistics in a dynamic bar chart. The system is optimized with multi-threading to ensure smooth video display alongside intensive detection tasks.
資工二B 411203380 張庭嘉
---

## Features

* **即時物件偵測 (Real-time Object Detection):** 利用 YOLOv8n 模型從即時攝像頭畫面中高效準確地偵測物件。
* **優化效能 (Optimized Performance):** 採用多執行緒技術，將物件偵測推論在獨立執行緒中運行，防止主影像顯示卡頓。
* **可配置偵測頻率 (Configurable Detection Frequency):** 物件偵測推論約每 **1.5 秒**執行一次。
* **自動數據記錄 (Automated Data Logging):** 自動將偵測到的物件類別和數量連同時間戳記錄到 `results.csv` 文件中。
* **動態數據視覺化 (Dynamic Data Visualization):** 在 Matplotlib 長條圖中顯示即時物件計數統計，約每 **2.5 秒**刷新一次。
* **使用者互動 (User Interaction):**
    * 按下 `q` 或 `Esc` 鍵可退出應用程式。
    * 按下 `s` 鍵可保存當前帶有偵測結果的畫面。

---

## Technical Stack (技術堆疊)

* **Python:** 整個系統的核心程式語言。
* **`ultralytics`:** 提供 YOLOv8 模型，用於最先進的物件偵測。
* **`opencv-python` (cv2):** 用於影像捕捉、幀處理和顯示帶有偵測結果的即時畫面。
* **`matplotlib`:** 支援即時數據視覺化圖表。
* **`threading` & `queue`:** Python 內建模組，用於並行執行和執行緒安全的數據交換，對效能優化至關重要。
* **`csv` & `os`:** 標準庫，用於文件 I/O 操作 (CSV 記錄) 和文件系統互動。

---

## Algorithms & Principles (演算法與原理)

### Object Detection: YOLOv8 (物件偵測：YOLOv8)

本系統使用 Ultralytics 系列的 **YOLOv8n** (nano) 模型。YOLO (You Only Look Once) 是一種單階段物件偵測演算法，以其速度和準確性而聞名。它直接從整個圖像中預測邊界框和類別概率，使其在即時應用中非常高效。

### Concurrency: Multi-threading (並行性：多執行緒)

為了確保流暢的使用者體驗，應用程式採用了三個主要執行緒：

1.  **主執行緒 (Main Thread):** 負責連續的攝像頭幀讀取 (`cv2.VideoCapture`)、即時影像顯示 (`cv2.imshow`) 和鍵盤輸入管理。它還處理來自推論執行緒的結果，並更新繪圖執行緒的數據。
2.  **推論執行緒 (Inference Thread):** 專門用於運行計算密集型的 YOLOv8 模型。它處理從主執行緒接收到的幀，並大約每 **1.5 秒**執行一次物件偵測。
3.  **繪圖執行緒 (Plotting Thread):** 負責持續更新和顯示 Matplotlib 長條圖，反映累積的物件計數。此執行緒約每 **2.5 秒**刷新一次圖表。

執行緒之間的數據傳輸（幀到推論，結果返回主執行緒）是通過**執行緒安全的 `queue.Queue`** 物件進行管理，以防止競爭條件並確保數據完整性。

### Data Visualization: Matplotlib Interactive Mode (數據視覺化：Matplotlib 互動模式)

`matplotlib.pyplot` 的互動模式 (`plt.ion()`) 允許圖表在不關閉視窗的情況下動態更新，提供偵測統計的即時視覺化表示。

---

## Setup and Installation (設定與安裝)

### Prerequisites (先決條件)

* Python 3.x
* 已連接的攝像頭

### Installation Steps (安裝步驟)

1.  **Clone the repository (如果適用):**
    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```
    (如果您沒有儲存庫，可以跳過此步驟，直接創建 Python 文件。)

2.  **Create a virtual environment (推薦):**
    ```bash
    python -m venv venv
    # 在 Windows 上
    venv\Scripts\activate
    # 在 macOS/Linux 上
    source venv/bin/activate
    ```

3.  **Install the required packages (安裝所需套件):**
    ```bash
    pip install ultralytics opencv-python matplotlib numpy
    ```
    *注意: 對於 Matplotlib，如果您遇到 `TkAgg` 或 `Qt5Agg` 後端未找到的問題，您可能需要安裝 `tkinter` (通常包含在 Python 中，但有時需要單獨的系統套件) 或 `PyQt5` (`pip install PyQt5`)。*

4.  **Download the YOLOv8n model (下載 YOLOv8n 模型):**
    `ultralytics` 庫在第一次使用時，如果 `yolov8n.pt` 模型不在您的工作目錄中，將會自動下載。首次運行時請確保網路連接。

---

## Usage (使用方式)

1.  **Save the code (保存程式碼):** 將提供的 Python 腳本保存為 `main.py` (或任何您喜歡的名稱) 在您的專案目錄中。
2.  **Run the script (運行腳本):**
    ```bash
    python main.py
    ```

### Interactions (互動):

* 帶有即時偵測結果的攝像頭畫面將出現在 `cv2` 視窗中。
* 一個獨立的 `matplotlib` 視窗將顯示物件計數長條圖。
* 按下鍵盤上的 **`q`** 或 **`Esc`** 鍵 (當 `cv2` 視窗處於活動狀態時) 可優雅地關閉應用程式。
* 按下 **`s`** 鍵可將當前帶有偵測結果的畫面保存到 `captured_frames` 目錄中。

---

## CSV Output (CSV 輸出)

偵測結果將記錄到腳本所在目錄的 `results.csv` 文件中。CSV 文件將包含以下欄位：

* `Timestamp`: 偵測的日期和時間。
* `Class`: 偵測到的物件類別名稱 (例如：'person', 'car')。
* `Count`: 該時間間隔內該特定類別偵測到的物件數量。

---

## Project Structure (專案結構範例)
