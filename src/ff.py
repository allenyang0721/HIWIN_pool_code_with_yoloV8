import cv2

# 開啟攝影機（攝影機ID可依需要更改）
cap = cv2.VideoCapture(3)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 無法取得影像")
        break

    # 在畫面上 (1000, 700) 標記紅色圓點
    # cv2.circle(frame, (700, 200), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.circle(frame, (700, 70), radius=5, color=(0, 0, 255), thickness=-1)
    # 顯示影像
    cv2.imshow("Red Point at (1000, 700)", frame)

    # 按下 ESC 鍵退出
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
