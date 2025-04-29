import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 建立 RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# 啟用彩色影像串流
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

# 啟動 pipeline
pipeline.start(config)

print("按下空白鍵以拍照並儲存圖片，按下 ESC 鍵退出程式")

# 創建儲存圖片的資料夾
save_dir = 'captured_images'
os.makedirs(save_dir, exist_ok=True)

img_counter = 0

try:
    while True:
        # 等待新的 frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # 將影像轉成 numpy 陣列
        color_image = np.asanyarray(color_frame.get_data())

        # 顯示影像
        cv2.imshow('RealSense Camera', color_image)

        key = cv2.waitKey(1)

        if key == 27:  # 按下 ESC 鍵
            print("結束程式")
            break
        elif key == 32:  # 按下空白鍵
            img_name = os.path.join(save_dir, f"photo_{img_counter:03d}.png")
            cv2.imwrite(img_name, color_image)
            print(f"已儲存：{img_name}")
            img_counter += 1

finally:
    # 關閉 pipeline 與視窗
    pipeline.stop()
    cv2.destroyAllWindows()