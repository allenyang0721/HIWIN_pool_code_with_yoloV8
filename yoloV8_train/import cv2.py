import cv2

print("🔍 掃描可用攝影機...")
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.read()[0]:
        print(f"✅ Camera index {i} 可用")
        cap.release()
    else:
        print(f"❌ Camera index {i} 不可用")
