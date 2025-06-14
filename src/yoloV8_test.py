import os
import math
import json
import time
import socket
import cv2
import numpy as np
import serial
from Coordinate_mapping_combined import (
    first_calibration, second_calibration_new
)
from ultralytics import YOLO

# 請確保下面這些函式在同一個檔案中可用
# from your_module import run_once_and_render, hit_ball

# 1️⃣ 載入 YOLO 模型（路徑改成你的 .pt 檔案位置）
model = YOLO(r"C:\Users\Gillion-BennyWinNB\Desktop\2025HIWIN_poolball\models\956PT_0428_best.pt")

def detect_and_render(frame):
    """
    對單張影像做 YOLO 偵測，並回傳預測結果物件。
    """
    results = model.predict(
        source=frame,
        save=False,
        imgsz=640,
        conf=0.5
    )
    return results[0]  # 取第一張（此例只有一張）

def main():
    # 打開攝影機
    cap = cv2.VideoCapture(3)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 15)

    print("🚀 啟動中，按下 SPACE 拍照並開始偵測／決策；ESC 離開。")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 無法讀取攝影機影像")
            break

        # 即時畫面
        cv2.imshow("Live View", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("結束")
            break

        elif key == 32:  # SPACE
            # 1. 拍照（直接取得當下 frame）
            snapshot = frame.copy()

            # 2. YOLO 偵測（你可自行視覺化 detect_result.show()）
            detect_result = detect_and_render(snapshot)

            # （選擇性）顯示偵測框結果
            detect_result.show()
            cv2.waitKey(0)
            cv2.destroyWindow("YOLO")

            # 3. 執行你的撞球決策與繪圖邏輯
            #    run_once_and_render 回傳 (out_img, params)
            out_img, params = run_once_and_render(snapshot)

            # 4. 印出參數，並把路徑圖存檔
            print("決策參數：", params)
            cv2.imwrite("out.png", out_img)

            # 5. 根據參數呼叫打擊函式
            hit_ball()

            # 6. （選擇性）顯示最終路徑圖
            cv2.imshow("決策後路徑", out_img)
            cv2.waitKey(0)
            cv2.destroyWindow("決策後路徑")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
