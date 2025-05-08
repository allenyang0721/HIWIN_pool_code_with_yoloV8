import cv2
import os

# 初始化攝影機（0 通常是預設攝影機）
cap = cv2.VideoCapture(2)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 6)

print("按下空白鍵以拍照並儲存圖片，按下 ESC 鍵退出程式")

# 建立儲存圖片資料夾
save_dir = r"C:\Users\Gillion-BennyWinNB\Desktop\2025HIWIN_poolball\han_test"
os.makedirs(save_dir, exist_ok=True)

img_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("無法讀取影像")
        break

    cv2.imshow('Webcam', frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC 鍵
        print("結束程式")
        break
    elif key == 32:  # 空白鍵
        img_name = os.path.join(save_dir, f"han_{img_counter:03d}.png")
        cv2.imwrite(img_name, frame)
        print(f"已儲存：{img_name}")
        img_counter += 1

# 釋放資源
cap.release()
cv2.destroyAllWindows()
