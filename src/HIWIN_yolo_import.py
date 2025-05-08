import math
import numpy as np
import cv2
import os
import time
import json
from ultralytics import YOLO

# ------- ⚡ 參數設定 ⚡ -------
BALL_RADIUS = 28
COLLISION_RADIUS = 58
FAKEBALL_OFFSET = BALL_RADIUS * 2
PROJECT_ROOT = r"C:\Users\Gillion-BennyWinNB\Desktop\2025HIWIN_poolball"
SAVE_PATH   = os.path.join(PROJECT_ROOT, "output", "pool_analysis.png")
ROUTE_PATH  = os.path.join(PROJECT_ROOT, "output", "route_data.json")
os.makedirs(os.path.dirname(ROUTE_PATH), exist_ok=True)

# ------- ⚡ 拍照階段：使用 D435 RGB 相機（OpenCV） -------
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 15)
print("📸 按 [空白鍵] 拍照, [ESC] 離開")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow('D435 RGB View', frame)
    key = cv2.waitKey(1)
    if key == 32:  # Space
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        IMG_PATH = f"{PROJECT_ROOT}\\take_images\\captured_{timestamp}.png"
        os.makedirs(os.path.dirname(IMG_PATH), exist_ok=True)
        cv2.imwrite(IMG_PATH, frame)
        print(f"✅ 拍照成功：{IMG_PATH}")
        break
    elif key == 27:  # ESC
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# ------- ⚡ YOLO 模型推論 ⚡ -------
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "956PT_0428_best.pt")
model = YOLO(MODEL_PATH)
results = model(IMG_PATH)
homo = cv2.imread(IMG_PATH)
img_h, img_w = homo.shape[:2]

# 解析 YOLO 偵測結果
detections = []
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0].item())
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        detections.append({'class_id': class_id, 'center': (cx, cy), 'xyxy': [x1,y1,x2,y2]})

# 分類球與洞口
mom = None
ball_dict = {}
holes = []
for det in detections:
    cid, pos = det['class_id'], det['center']
    if cid == 0:           mom = pos
    elif cid in [1,2,3]:   ball_dict[cid] = pos
    elif cid == 4:         holes.append(pos)

# 幾何函式
def crush_precise(a, b, obs):
    a, b = np.array(a), np.array(b)
    ab = b - a; ab_len = np.linalg.norm(ab)
    for c in obs:
        c = np.array(c)
        proj = np.dot(c-a, ab) / ab_len
        if 0 < proj < ab_len:
            dist = np.linalg.norm(c - (a + ab*(proj/ab_len)))
            if dist <= COLLISION_RADIUS:
                return True
    return False

def angle_from_downward_signed(center, target):
    dx, dy = target[0]-center[0], target[1]-center[1]
    rad = math.atan2(-dx, dy)
    deg = math.degrees(rad)
    return deg-360 if deg>180 else deg

def DOT(p):    cv2.circle(homo, (int(p[0]),int(p[1])), 5, (200,0,0), -1)
def line(a,b): cv2.line(homo, (int(a[0]),int(a[1])), (int(b[0]),int(b[1])), (150,70,130), 3)

# 找出要打的球
bid   = min(k for k in ball_dict if k in [1,2,3])
rball = ball_dict[bid]
obs   = [v for k,v in ball_dict.items() if k!=bid]
wball = mom

# ------- ⚡ 直擊 or 反彈 判斷 ⚡ -------
output_data = {}
if not crush_precise(wball, rball, obs):
    # 直擊
    print(f"✅ Ball{bid} 可直擊！")
    DOT(wball); DOT(rball)
    line(wball, rball)
    cue_angle = angle_from_downward_signed(wball, rball)
    output_data = {
        "type": "direct",
        "wball": wball,
        "rball": rball,
        "cue_angle": cue_angle
    }
else:
    # 嘗試反彈（保留你原本的反彈邏輯）
    print("嘗試反彈")
    valid_rebounds = []
    # ... 這裡請貼上原先 rebound 生成、篩選、評分程式 ...
    # 假設你最後選到 ana, bnb, h 並計算 incident_angle, exit_angle, cue_angle
    # 範例：
    # output_data = {
    #     "type": "rebound",
    #     "wball": wball,
    #     "fakeball": ana,
    #     "rebound_point": bnb,
    #     "hole": h,
    #     "cue_angle": cue_angle,
    #     "incident_angle": incident_angle,
    #     "exit_angle": exit_angle
    # }
    # 若無可行反彈，僅標示
    if not valid_rebounds:
        print("🚫 無可行反彈路徑")

# 輸出 JSON
with open(ROUTE_PATH, 'w') as f:
    json.dump(output_data, f, indent=2)

# 儲存影像
cv2.putText(homo, f"Cue Angle: {output_data.get('cue_angle',0):.1f}°", (20,100),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
cv2.imwrite(SAVE_PATH, homo)
print(f"✅ 圖片儲存：{SAVE_PATH}")
cv2.imshow('poolball', homo)
cv2.waitKey(0)
cv2.destroyAllWindows()
