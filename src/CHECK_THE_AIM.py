import socket
import cv2
import numpy as np
import time
import math

# === 📌 攝影機與模板設定 ===
template_path = r"C:\Users\Gillion-BennyWinNB\Desktop\2025HIWIN_poolball\second_data\0507gay.png"
cap_id = 2  # 攝影機ID

def get_center():
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)[1]
    w, h = template.shape[::-1]

    cap = cv2.VideoCapture(cap_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    for _ in range(30):
        cap.read()
        time.sleep(0.03)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise Exception("❌ 錄影失敗")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result = cv2.matchTemplate(binary, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    center = (max_loc[0] + w // 2, max_loc[1] + h // 2)
    return center

def send_to_arm(command_str):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", 9000))
    sock.sendall(command_str.encode())
    sock.close()
    print(f"🚀 傳送手臂指令：{command_str}")

def test_axis(axis_name):
    print(f"\n==============================")
    print(f"🔬 開始測試 {axis_name} 軸單位移動像素差")
    print(f"==============================")

    base_x, base_y, z = -120, -88, 100
    a, b, c = 180, 0, -92
    distances = []

    for i in range(5):
        # 發送初始位置
        pos_1 = f"[{base_x}, {base_y}, {z}, {a}, {b}, {c}]"
        send_to_arm(pos_1)
        time.sleep(2.5)
        center_1 = get_center()

        # 移動對應軸
        if axis_name == "X":
            base_x += 1
        elif axis_name == "Y":
            base_y += 1
        else:
            raise ValueError("軸名稱錯誤，只接受 'X' 或 'Y'")

        # 發送下一位置
        pos_2 = f"[{base_x}, {base_y}, {z}, {a}, {b}, {c}]"
        send_to_arm(pos_2)
        time.sleep(2.5)
        center_2 = get_center()

        dx = center_2[0] - center_1[0]
        dy = center_2[1] - center_1[1]
        dist = round(math.sqrt(dx**2 + dy**2), 2)
        print(f"第 {i+1} 次移動：ΔX = {dx}, ΔY = {dy}, 像素距離 = {dist}")
        distances.append(dist)

    avg_dist = round(sum(distances) / len(distances), 2)
    print(f"✅ {axis_name} 軸平均像素/單位：{avg_dist} pixels/unit")
    return avg_dist

# === 🚀 主程序：測 X 軸，再測 Y 軸 ===
avg_x = test_axis("X")
avg_y = test_axis("Y")

print("\n🎯 最終結果：")
print(f"   ➤ X軸：{avg_x} pixels/unit")
print(f"   ➤ Y軸：{avg_y} pixels/unit")
