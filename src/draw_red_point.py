import cv2

# 讀取白球模板（需為灰階、二值化）
template = cv2.imread(r"C:\Users\Gillion-BennyWinNB\Desktop\2025HIWIN_poolball\second_data\0507gay.png", cv2.IMREAD_GRAYSCALE)
template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
w, h = template.shape[::-1]

# 開啟攝影鏡頭
cap = cv2.VideoCapture(3)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("🚀 啟動成功！請按空白鍵進行辨識，ESC 鍵離開。")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 錄影失敗")
        break

    # 顯示原始畫面
    cv2.imshow("Live", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC 鍵離開
        print("🔚 結束程式")
        break
    elif key == 32:  # 空白鍵（ASCII 32）
        print("📸 進行辨識...")

        # 灰階 + 二值化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 模板匹配
        result = cv2.matchTemplate(binary, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # 標示匹配區域與中心點
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        center_x = top_left[0] + w // 2
        center_y = top_left[1] + h // 2
        center = (center_x, center_y)
        print(f"🎯 母球中心點座標：{center}")

        # 畫框與標記
        output_frame = frame.copy()
        cv2.circle(output_frame, (775, 140), radius=5, color=(0, 0, 255), thickness=-1)
        cv2.rectangle(output_frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.circle(output_frame, center, 5, (0, 0, 255), -1)
        cv2.putText(output_frame, f"{center}", (center_x + 10, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        

        # 顯示結果
        cv2.imshow("Match Result", output_frame)

cap.release()
cv2.destroyAllWindows()
