import math
import numpy as np
import cv2
import os

# ------- ⚡ 參數設定 ⚡ -------
BALL_RADIUS =28
COLLISION_RADIUS = 58  
FAKEBALL_OFFSET = BALL_RADIUS * 2
PROJECT_ROOT=r"C:\Users\Gillion-BennyWinNB\Desktop\2025HIWIN_poolball"#只需要把這行改成這個資料夾的絕對路徑即可
SAVE_PATH = os.path.join(PROJECT_ROOT, "output", "pool_analysis.png")
ROUTE_PATH = os.path.join(PROJECT_ROOT, "output", "route_data.json")
os.makedirs(os.path.dirname(ROUTE_PATH), exist_ok=True)

# ------- ⚡ 拍照階段：使用 D435 RGB 相機（OpenCV） -------
import time
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # 寬度，例如 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # 高度，例如 720
cap.set(cv2.CAP_PROP_FPS, 15)        
print("📸 按 [空白鍵] 拍照, [ESC] 離開")
#IMG_PATH = r"C:\Users\Gillion-BennyWinNB\Desktop\2025HIWIN_poolball\take_images\captured_20250504_213321.png"#改成圖片的絕對路徑
IMG_PATH=None
take="take_images"
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow('D435 RGB View', frame)
    key = cv2.waitKey(1)
    if key == 32:  # Space
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        IMG_PATH = f"{PROJECT_ROOT}\{take}\captured_{timestamp}.png"
        cv2.imwrite(IMG_PATH, frame)
        print(f"✅ 拍照成功：{IMG_PATH}")
        break
    elif key == 27:# ESC
        cap.release()
        cv2.destroyAllWindows()
        exit()
cap.release()
cv2.destroyAllWindows()

# ------- ⚡ YOLO 模型推論 ⚡ -------
from ultralytics import YOLO
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "956PT_0428_best.pt")
# IMG_PATH 由拍照階段自動指定
model = YOLO(MODEL_PATH)
results = model(IMG_PATH)
homo = cv2.imread(IMG_PATH)
img_h, img_w = homo.shape[:2]

# 直接使用 YOLO 模型輸出的結果

detections = []
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0].item())
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        detections.append({
            'class_id': class_id,
            'xyxy': [x1, y1, x2, y2],
            'center': (cx, cy)
        })

# ------- ⚡ 球類別分類 ⚡ -------
mom = None
ball_dict = {}
holes = []

for det in detections:
    class_id = det['class_id']
    cx, cy = det['center']
    pos = (cx, cy)
    if class_id == 0:
        mom = pos
    elif class_id in [1, 2, 3]:
        ball_dict[class_id] = pos
    elif class_id == 4:
        holes.append(pos)

# ------- ⚡ 幾何函式 ⚡ -------
def getLong(a, b):
    return math.hypot(a, b)

def vector(a, b):
    return [b[0] - a[0], b[1] - a[1]]

def fakeball(ball, hole, offset=FAKEBALL_OFFSET):
    dx = hole[0] - ball[0]
    dy = hole[1] - ball[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return ball
    scale = offset / length
    fx = ball[0] - dx * scale
    fy = ball[1] - dy * scale
    return (int(fx), int(fy))

def getAngle(a, b, c):
    AB = np.array(b) - np.array(a)
    BC = np.array(c) - np.array(b)
    lab = np.linalg.norm(AB)
    lbc = np.linalg.norm(BC)
    if lab == 0 or lbc == 0:
        return 999
    cos_theta = np.dot(AB, BC) / (lab * lbc)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return math.acos(cos_theta) * 180 / math.pi

def crush_precise(a, b, c_list):
    a = np.array(a)
    b = np.array(b)
    ab = b - a
    ab_len = np.linalg.norm(ab)
    for c in c_list:
        c = np.array(c)
        ac = c - a
        proj = np.dot(ac, ab) / ab_len
        if proj < 0 or proj > ab_len:
            continue
        proj_point = a + (ab * proj / ab_len)
        dist = np.linalg.norm(c - proj_point)
        if dist <= COLLISION_RADIUS:
            return True
    return False

def crush_one_precise(a, b, c):
    return crush_precise(a, b, [c])

def DOT(a):
    cv2.circle(homo, (int(a[0]), int(a[1])), 5, (200, 0, 0), -1)

def line(a, b):
    a = (int(a[0]), int(a[1]))
    b = (int(b[0]), int(b[1]))
    cv2.line(homo, a, b, (150, 70, 130), 3)

def rebound(fake, wall_bounds):
    margin = BALL_RADIUS
    min_x, max_x, min_y, max_y = wall_bounds
    dx = fake[0] - wball[0]
    dy = fake[1] - wball[1]
    total = math.hypot(dx, dy)
    if total == 0:
        mid = fake
    else:
        ratio = 0.5  # 取一半
        mid = (wball[0] + dx * ratio, wball[1] + dy * ratio)

    wall_points = [
        (mid[0], min_y + margin),     # 上牆
        (mid[0], max_y - margin),     # 下牆
        (min_x + margin, mid[1]),     # 左牆
        (max_x - margin, mid[1])      # 右牆
    ]

    # 過濾距離袋口太近的反彈點
    wall_points = [pt for pt in wall_points if all(np.linalg.norm(np.array(pt) - np.array(h)) >= BALL_RADIUS * 2 for h in holes)]
    return wall_points

def fakeball_list(kid, hole):
    return [fakeball(kid, h) for h in hole]

def get_table_bounds(holes):
    xs = [p[0] for p in holes]
    ys = [p[1] for p in holes]
    return min(xs), max(xs), min(ys), max(ys)

# ------- ⚡ 擇一目標球路徑 + 評分 ⚡ -------
# 🧩 打反彈時會遇到的限制條件整理：
# 1. wball → bnb（母球 → 反彈點）不能有障礙
# 2. bnb → ana（反彈點 → 假想球）不能有障礙
# 3. ana → h（假想球 → 洞口）不能有障礙
# 4. ana → rball（假想球 → 目標球）不能被提前碰撞
# 5. rball → h（目標球 → 洞口）不能有障礙
# 6. ∠ana, rball, h ≤ 90°（目標球往洞角方向不能偏離太大）
# 7. ∠bnb, ana, h ≤ 45°（反彈線不能過斜）
# 8. ∠入射與出射應盡量接近
# 畫出淺綠色牆面
min_x, max_x, min_y, max_y = get_table_bounds(holes)
cv2.line(homo, (min_x, min_y), (max_x, min_y), (144, 238, 144), 2)  # 上牆
cv2.line(homo, (min_x, max_y), (max_x, max_y), (144, 238, 144), 2)  # 下牆
cv2.line(homo, (min_x, min_y), (min_x, max_y), (144, 238, 144), 2)  # 左牆
cv2.line(homo, (max_x, min_y), (max_x, max_y), (144, 238, 144), 2)  # 右牆
bid = min([k for k in ball_dict.keys() if k in [1, 2, 3]])
rball = ball_dict[bid]
obs = [v for k, v in ball_dict.items() if k != bid]
wball = mom

crush_one_precise(mom,mom, mom)

best_path = None
best_score = float('inf')
valid_rebounds = []

hol = [h for h in holes if 0 <= getAngle(wball, rball, h) <= 90]
for h in hol:
    fake = fakeball(rball, h)
    if crush_precise(wball, fake, obs):
        #print("🚫 母球到假想球 被障礙擋住")
        continue
    # 忽略母球提前打到目標球，不當作碰撞失敗
    # if crush_one_precise(wball, fake, rball):
    #     print("🚫 母球提前碰到目標球")
    #     continue
    if crush_precise(fake, h, obs):
        #print("🚫 假想球到洞口路徑被障礙擋住")
        continue
    if crush_precise(rball, h, obs):
        #print("🚫 目標球到洞口之間有障礙球")
        # 嘗試組合球代替直擊
        continue  # 確保目標球到洞口這段不被擋住
    if crush_precise(fake, rball, obs):
        #print("🚫 假想球到目標球 被障礙擋住")
        continue  # 假想球擊中目標球路徑不得被擋住
    if crush_precise(wball, rball, obs):
        #print("🚫 母球到目標球之間有障礙球")
        continue  # ⛔ 額外補強：母球 → 目標球也不得被阻擋

    
    if getAngle(wball, rball, h) > 90:
        #print("🚫 直擊夾角大於90度")
        continue

    angle_score = getAngle(wball, fake, rball)
    dist_score = np.linalg.norm(np.array(wball) - np.array(fake))
    score = angle_score + dist_score * 0.05

    if score < best_score:
        best_score = score
        best_path = (h, fake)

if best_path:
    h, fake = best_path
    print(f"✅ Ball{bid} 可直擊 (最佳袋口)")
    DOT(wball)
    DOT(rball)
    for o in obs:
        DOT(o)
    for ho in holes:
        DOT(ho)
    line(h, fake)
    line(wball, fake)
    cv2.circle(homo, fake, BALL_RADIUS, (200, 100, 0), 2)
    DOT(fake)
else:
    print(f"嘗試組合球")
    if len(obs) >= 1:
        best_combo = None
        best_combo_score = float('inf')
        for mid in obs:
            mixf1 = fakeball(mid, h)
            mixf2 = fakeball(rball, mixf1)
            if not crush_precise(wball, mixf2, obs) and not crush_precise(mixf2, rball, obs) and not crush_precise(rball, mixf1, obs) and not crush_precise(mid, h, obs) and not crush_precise(rball, mid, obs):
                angle_score = getAngle(wball, mixf2, rball) + getAngle(rball, mixf1, mid) 
                dist_score = np.linalg.norm(np.array(wball) - np.array(mixf2)) + \
                             np.linalg.norm(np.array(mixf2) - np.array(rball)) + \
                             np.linalg.norm(np.array(rball) - np.array(mixf1)) + \
                             np.linalg.norm(np.array(mixf1) - np.array(mid)) + \
                             np.linalg.norm(np.array(mid) - np.array(h))
                if getAngle(wball, mixf2, rball) > 90 or getAngle(rball, mixf1, mid) > 90:
                    continue
                score = angle_score + dist_score * 0.01
                if score < best_combo_score:
                    best_combo_score = score
                    best_combo = (mixf1, mixf2, mid)
        if best_combo:
            mixf1, mixf2, mid = best_combo
            cv2.circle(homo, mixf1, BALL_RADIUS, (0, 100, 255), 2)
            cv2.circle(homo, mixf2, BALL_RADIUS, (0, 100, 255), 2)
            DOT(mixf2)
            DOT(mixf1)
            line(wball, mixf2)
            line(mixf2, rball)
            line(rball, mixf1)
            line(mixf1, mid)
            line(mid, h)
        else:
            print("🚫 組合球路徑也被阻擋")
    print(f"❌ Ball{bid} 無法直擊，嘗試反彈")
    rebf = fakeball_list(rball, holes)
    # 正確產生所有反彈點（每個假想球對應 8 點）
    rebb = []
    wall_bounds = get_table_bounds(holes)
    for fb in rebf:
        rebb.extend(rebound(fb, wall_bounds))

    for h in holes:
        for fb in rebf:
            wall_bounds = get_table_bounds(holes)
            all_rebounds = rebound(fb, wall_bounds)
            for bnb in all_rebounds:
            
                ana = fb
                if not crush_precise(wball, bnb, obs) and not crush_precise(bnb, ana, obs) and not crush_precise(ana, h, obs):
                    if not crush_one_precise(ana, bnb, rball) and not crush_precise(ana, h, obs):
                        angle = getAngle(ana, rball, h)
                        if angle > 90:
                            continue  # 排除命中目標球後，目標球往袋口方向偏離太大者
                        if angle <= 10 and getAngle(bnb, ana, h) <= 45 and getAngle(ana, rball, h) <= 90:
                            valid_rebounds.append((angle, (ana, bnb, h)))

    if valid_rebounds:
        best_rebound = None
        best_rebound_score = float('inf')
        for angle, (ana, bnb, h) in valid_rebounds:
            total_dist = np.linalg.norm(np.array(wball) - np.array(bnb)) + \
                         np.linalg.norm(np.array(bnb) - np.array(ana)) + \
                         np.linalg.norm(np.array(ana) - np.array(rball))
            score = angle + total_dist * 0.01
            if score < best_rebound_score:
                best_rebound_score = score
                best_rebound = (angle, (ana, bnb, h))

        angle, (ana, bnb, h) = best_rebound
        print(f"⭐ 最佳反彈角度 {angle:.2f}°")
        cv2.circle(homo, (int(ana[0]), int(ana[1])), BALL_RADIUS, (0, 255, 255), 2)
        DOT(ana)
        DOT(bnb)
        line(ana, bnb)
        line(rball, ana)
        line(h, rball)
        line(wball, bnb)
        # 畫出法線方向（藍色）
        dx = ana[0] - bnb[0]
        dy = ana[1] - bnb[1]
        if abs(dx) < abs(dy):  # 撞垂直牆（左右）→ 法線為水平
            normal_dir = np.array([0, 1])  # 垂直方向
        else:  # 撞水平牆（上下）→ 法線為垂直
            normal_dir = np.array([1, 0])  # 水平方向
        normal_end = (int(bnb[0] + normal_dir[0] * 50), int(bnb[1] + normal_dir[1] * 50))
        cv2.arrowedLine(homo, (int(bnb[0]), int(bnb[1])), normal_end, (255, 0, 0), 2, tipLength=0.2)

import json

# ------- ⚡ 入射角 / 出射角計算（法線計算） ⚡ -------
def compute_incident_angle(wball, bnb, ana):
    # 入射角 = 母球 → 反彈點 與 法線 的夾角（假設牆面法線與 bnb → ana 垂直）
    incoming = np.array(bnb) - np.array(wball)
    normal = np.array(ana) - np.array(bnb)
    normal_len = np.linalg.norm(normal)
    if normal_len == 0:
        return 999
    normal = normal / normal_len
    incoming_len = np.linalg.norm(incoming)
    if incoming_len == 0:
        return 999
    incoming = incoming / incoming_len
    dot_product = np.dot(incoming, normal)
    angle = math.acos(np.clip(dot_product, -1.0, 1.0))
    return math.degrees(angle)

def compute_exit_angle(ana, rball, h):
    # 出射角 = 反彈點 → 假想球 → 目標球方向 與 法線的夾角
    outgoing = np.array(rball) - np.array(ana)
    normal = np.array(ana) - np.array(bnb)
    normal_len = np.linalg.norm(normal)
    if normal_len == 0:
        return 999
    normal = normal / normal_len
    outgoing_len = np.linalg.norm(outgoing)
    if outgoing_len == 0:
        return 999
    outgoing = outgoing / outgoing_len
    dot_product = np.dot(outgoing, normal)
    angle = math.acos(np.clip(dot_product, -1.0, 1.0))
    return math.degrees(angle)

incident_angle = None
exit_angle = None
if not best_path:
    incident_angle = compute_incident_angle(wball, bnb, ana)
    exit_angle = compute_exit_angle(ana, rball, h)


def angle_from_downward_signed(center, target):
    """
    以母球為中心，6點鐘方向為 0 度。
    順時針為正（0 ~ 179.9°），逆時針為負（0 ~ -179.9°）。
    """
    dx = target[0] - center[0]
    dy = target[1] - center[1]
    rad = math.atan2(dx, dy)  # 以 y 軸正方向（6點）為基準
    deg = math.degrees(rad)
    # 限制角度範圍在 (-180, 180]
    if deg > 180:
        deg -= 360
    return deg

if best_path:
    cue_angle = angle_from_downward_signed(wball, fake)
else:    
    target_point = fake if best_path else bnb # 目標點是反彈點還是假想球
    if best_path:
        cue_angle = angle_from_downward_signed(wball, target_point)
    else:
        cue_angle = angle_from_downward_signed(wball, bnb)
        target_point = fake if best_path else bnb



# ------- ⚡ 導出擊球路徑 JSON ⚡ -------
output_data = {
    # "bid": bid,
    # "type": "direct" if best_path else "rebound",
    # "rball": rball,
    # "hole": h,
    # "fakeball": fake if best_path else ana,
    # "rebound_point": None if best_path else bnb,
    # "angle": None if best_path else angle,
    # "incident_angle": incident_angle,
    # "exit_angle": exit_angle,
    "wball": wball,
    "cue_angle": cue_angle
}

with open(ROUTE_PATH, 'w') as f:
    json.dump(output_data, f, indent=2)
cv2.circle(homo, mom, BALL_RADIUS, (255, 255, 255), 2)
for k, v in ball_dict.items():
    cv2.circle(homo, v, BALL_RADIUS, (0, 0, 255) if v == rball else (255, 0, 255), 2)

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

cv2.putText(homo, os.path.basename(IMG_PATH), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(homo, f"Ball{bid}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 255), 2)
cv2.putText(homo, f"Cue Angle: {cue_angle:.1f}°", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2) 
cv2.imwrite(SAVE_PATH, homo)
if os.path.exists(SAVE_PATH):
    print(f"✅ 圖片儲存成功：{SAVE_PATH}")
else:
    print(f"❌ 圖片儲存失敗，請確認資料夾是否存在")

    # if incident_angle is not None:
    #     cv2.putText(homo, f"Incident: {incident_angle:.1f}°", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    # if exit_angle is not None:
    #     cv2.putText(homo, f"Exit: {exit_angle:.1f}°", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
print(f"母球座標:{mom}")#母球座標
print(f"Cue Angle: {cue_angle:.4f}°")
cv2.imshow('poolball', homo)
cv2.waitKey(0)
cv2.destroyAllWindows()

