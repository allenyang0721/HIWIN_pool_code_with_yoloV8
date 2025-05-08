from ultralytics import YOLO
import cv2

def capture_and_detect():
    # 1️⃣ 載入模型
    model = YOLO(r"C:\Users\Gillion-BennyWinNB\Desktop\2025HIWIN_poolball\models\956PT_0428_best.pt")

    # 2️⃣ 開啟鏡頭
    cap = cv2.VideoCapture(2)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ 無法讀取攝影機影像")
        return None, None, None

    # 3️⃣ YOLO 偵測
    results = model.predict(source=frame, save=False, imgsz=640, conf=0.5)
    result = results[0]

    # 4️⃣ 尋找類別 0 的第一個目標
    for box in result.boxes:
        class_id = int(box.cls[0])
        if class_id == 0:
            x1, y1, x2, y2 = box.xyxy[0]
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
            print(f"✅ 類別0中心座標：x={x_center}, y={y_center}")
            return frame, x_center, y_center

    print("⚠️ 未偵測到類別0")
    return frame, None, None

# 🔁 呼叫函式
image, x, y = capture_and_detect()

# ✅ 顯示拍攝影像（可選）
if image is not None:
    cv2.imshow("Captured Frame", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
