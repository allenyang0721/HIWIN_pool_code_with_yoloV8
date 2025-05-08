import socket
import cv2
import numpy as np
import math
from Coordinate_mapping_combined import first_calibration,second_calibration_new

# ---------- 🏁 母球座標輸入 ----------
x = int(input("請輸入母球 X 座標："))
y = int(input("請輸入母球 Y 座標："))
angle_on_image = float(input("請輸入母球角度："))



# ---------- 🧪 一次校正 ----------
a, b, _, _, _, c_int = first_calibration(x, y,angle_on_image)  # ✅ 只傳 x, y 即可
first_result = f"[{a}, {b}, 100, 180, 0, {c_int}]"


# 發送一次校正結果至手臂
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("127.0.0.1", 9000))
sock.sendall(first_result.encode())
sock.close()
print("✅ 一次校正完成，機械手臂移動至第一次校正位置")

# ---------- 📷 開啟鏡頭，取得 cvx, cvy ----------
template = cv2.imread(r"C:\Users\Gillion-BennyWinNB\Desktop\2025HIWIN_poolball\second_data\0507gay.png", cv2.IMREAD_GRAYSCALE)
template = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)[1]
w, h = template.shape[::-1]

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("🚀 按下空白鍵進行二次校正辨識（ESC 離開）")
cvx, cvy = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 錄影失敗")
        break

    cv2.imshow("Live", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        print("🔚 離開辨識")
        break
    elif key == 32:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = cv2.matchTemplate(binary, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        cvx = max_loc[0] + w // 2
        cvy = max_loc[1] + h // 2
        print(f"🎯 辨識到中心座標：({cvx}, {cvy})")
        break

cap.release()
cv2.destroyAllWindows()

if cvx is None or cvy is None:
    print("⚠️ 無法取得中心座標，流程中止")
    exit()

# ---------- 🧪 二次校正 ----------
x2, y2, _, _, _, c2_int= second_calibration_new(a, b, cvx, cvy, angle_on_image)
second_result = f"[{x2}, {y2}, 80, 180, 0, {c_int}]"

# 發送二次校正結果至手臂
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("127.0.0.1", 9000))
sock.sendall(second_result.encode())
sock.close()
print("✅ 二次校正完成，機械手臂移動至精修位置")
