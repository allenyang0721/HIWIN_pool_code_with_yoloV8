import math
import numpy as np
import cv2
import os
import socket
import time
import threading  # 新增：執行緒模組
from Coordinate_mapping_combined import first_calibration, second_calibration_new
import serial
import json
from ultralytics import YOLO

# ===== 新增：Preview Thread，從程式一開始就持續顯示相機畫面 =====
cap = cv2.VideoCapture(3)  # 改為3號鏡頭
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 15)

# Event 用於通知停止預覽
stop_preview = threading.Event()
#決策結果畫面
decision_frame = None


def preview_loop():
    global decision_frame
    while not stop_preview.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow('Camera Preview', frame)
        if decision_frame is not None:
            cv2.imshow('Decision Result', decision_frame)
        cv2.waitKey(1) & 0xFF
        # 讓執行緒不要佔滿 CPU
        time.sleep(0.01)


# 啟動 daemon 執行緒
threading.Thread(target=preview_loop, daemon=True).start()

# ------- ⚡ 參數設定 ⚡ -------
BALL_RADIUS = 28
COLLISION_RADIUS = 58
FAKEBALL_OFFSET = BALL_RADIUS * 2
PROJECT_ROOT = r"C:\Users\Gillion-BennyWinNB\Desktop\2025HIWIN_poolball"
SAVE_PATH = os.path.join(PROJECT_ROOT, "output", "pool_analysis.png")
ROUTE_PATH = os.path.join(PROJECT_ROOT, "output", "route_data.json")
os.makedirs(os.path.dirname(ROUTE_PATH), exist_ok=True)
coords_list = [-120, -88, 534, 180, 0, -92.5]


# ------- ⚡ 機械手臂回 Home（不會阻塞 Preview） -------
def move_arm_home():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", 9000))
    sock.sendall(str(coords_list).encode())
    time.sleep(5)
    print(f"手臂移動原點，座標：{coords_list}")
    sock.close()

# ------- ⚡ 拍照階段：只讀一次 frame 並儲存，不會阻塞 -------
def take_photo():
    ret, frame = cap.read()  # 單次讀取
    if ret:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        IMG_PATH = f"{PROJECT_ROOT}\\take_images\\captured_{timestamp}.png"
        cv2.imwrite(IMG_PATH, frame)
        print("✅ 拍照成功")
    else:
        print("❌ 拍照失敗，請檢查相機")
        # 若拍照失敗，可根據需求決定是否 raise Exception 或繼續
    return IMG_PATH



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

def angle_from_downward_signed(center, target):
    """
    以母球為中心，6點鐘方向為 0 度。
    順時針為正（0 ~ +179°），逆時針為負（0 ~ -179°）。
    """
    dx = target[0] - center[0]
    dy = target[1] - center[1]
    rad = math.atan2(-dx, dy)  # ⚠️ 將 dx 改為 -dx，使順時針為正
    deg = math.degrees(rad)
    if deg > 180:
        deg -= 358
    else:
        deg += 2
    return deg

def MAIN():
   
    # ------- ⚡ YOLO 模型推論 ⚡ -------
    move_arm_home()
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "956PT_0428_best.pt")
    model = YOLO(MODEL_PATH)
    IMG_PATH = take_photo()
    results = model(IMG_PATH)
    global homo
    homo = cv2.imread(IMG_PATH)
    
    
    # ------- ⚡ 決策判斷與路徑規劃 ⚡ -------
    # 以下插入您原本的直擊、組合球、反彈等決策邏輯與繪圖代碼
    # 例如：
    # detections = parse_detections(results)
    # wball, rball, holes, obs = classify_balls(detections)
    # best_path = compute_direct_path(...)
    # if not best_path: best_path = compute_combo_path(...)
    # if not best_path: best_rebound = compute_rebound_path(...)
    # 根據結果繪製路徑與法線示意到 homo
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
    global holes
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
    
    global wball
    wball = mom
    
    #crush_one_precise(mom,mom, mom)
    print("obs",obs)

    best_rebound=None
    best_path = None
    best_combo = None
    shrunk = None
    best_score = float('inf')
    valid_rebounds = []

    hol = [h for h in holes if 0 <= getAngle(wball, rball, h) <= 100]
    for h in hol:
        fake = fakeball(rball, h)
        if crush_precise(wball, fake, obs):
            print("🚫 母球到假想球 被障礙擋住")
            continue
        # 忽略母球提前打到目標球，不當作碰撞失敗
        # if crush_one_precise(wball, fake, rball):
            #print("🚫 母球提前碰到目標球")
        #     continue
        if crush_precise(fake, h, obs):
            print("🚫 假想球到洞口路徑被障礙擋住")
            continue
        if crush_precise(rball, h, obs):
            print("🚫 目標球到洞口之間有障礙球")
            # 嘗試組合球代替直擊
            continue  # 確保目標球到洞口這段不被擋住
        if crush_precise(fake, rball, obs):
            print("🚫 假想球到目標球 被障礙擋住")
            continue  # 假想球擊中目標球路徑不得被擋住
        if crush_precise(wball, rball, obs):
            print("🚫 母球到目標球之間有障礙球")
            continue  # ⛔ 額外補強：母球 → 目標球也不得被阻擋

        
        if getAngle(wball, fake, h) > 89 :
            print("🚫 直擊夾角大於89度")
            continue

        angle_score = getAngle(wball, fake, rball)
        dist_score = np.linalg.norm(np.array(wball) - np.array(fake))
        score = angle_score + dist_score * 0.1

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
            
            best_combo_score = float('inf')
            for mid in obs:
                mixf1 = fakeball(mid, h)
                mixf2 = fakeball(rball, mixf1)
                if not crush_precise(wball, mixf2, obs) :
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

            
            elif not best_combo and not best_path:
                if not crush_precise(wball, rball, obs):
                    print("直接打")
                    shrunk = True
                
                

                else:
                    print("🚫 組合球路徑也被阻擋")
                    print(f"❌ Ball{bid} 無法直擊，嘗試反彈")
                    rebf = fakeball_list(rball, holes)
                    # 正確產生所有反彈點（每個假想球對應 8 點）
                    rebb = []
                    best_rebound=None
                    wall_bounds = get_table_bounds(holes)
                    for fb in rebf:
                        rebb.extend(rebound(fb, wall_bounds))

                    for h in holes:
                        for fb in rebf:
                            wall_bounds = get_table_bounds(holes)
                            all_rebounds = rebound(fb, wall_bounds)
                            for bnb in all_rebounds:
                                ana = fb
                                if not crush_precise(wball, bnb, obs) and not crush_precise(bnb, ana, obs) :
                                
                                        angle = getAngle(ana, rball, h)
                                        if angle < 45:
                                            continue  # 排除命中目標球後，目標球往袋口方向偏離太大者
                                        if  getAngle(wball, bnb, ana) <= 150 :
                                            valid_rebounds.append((angle, (ana, bnb, h)))

                                        if valid_rebounds:
                                            best_rebound = None
                                            best_rebound_score = float('inf')
                                            for angle, (ana, bnb, h) in valid_rebounds:
                                                total_dist = np.linalg.norm(np.array(wball) - np.array(bnb)) + \
                                                            np.linalg.norm(np.array(bnb) - np.array(ana)) + \
                                                            np.linalg.norm(np.array(ana) - np.array(rball))
                                                score = total_dist 
                                                if score < best_rebound_score:
                                                    best_rebound_score = score
                                                    best_rebound = (angle, (ana, bnb, h))
                                                    continue
                    if best_rebound:
                        print(best_rebound)
                        angle, (ana, bnb, h) = best_rebound


                    
                    
                    cv2.circle(homo, (int(ana[0]), int(ana[1])), BALL_RADIUS, (0, 255, 255), 2)
                    DOT(ana)
                    DOT(bnb)
                    line(ana, bnb)
                    line(rball, ana)
                    line(h, rball)
                    line(wball, bnb)
                        
                

    print(rball)




    if best_path:
        cue_angle = angle_from_downward_signed(wball, fake)

    elif best_combo:
        cue_angle = angle_from_downward_signed(wball, mixf2)
    elif best_rebound:    
        cue_angle = angle_from_downward_signed(wball, bnb)
        if cue_angle > 180:
            cue_angle = cue_angle -3
        else: 
            cue_angle = cue_angle +3
    else:
        print("沒有最佳路徑,老子要直接打了")
        cue_angle = angle_from_downward_signed(wball, rball)



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



    # 將決策結果存到全域變數，讓 preview 顯示
    global decision_frame
    decision_frame = homo.copy()
    # 保存分析圖片
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    cv2.imwrite(SAVE_PATH, homo)
    print(f"✅ 分析結果已儲存到 {SAVE_PATH}")


    # ------- ⚡ 機械手臂校正 & 擊球操作 ⚡ -------
    # 1. 一次校正
    # 從 homo 影像的 wball 座標與 cue_angle 取 x, y, angle
    x, y = wball

    # 呼叫一次校正函式
    a, b, _, _, _, c_int = first_calibration(x, y, cue_angle)
    first_cmd = f"[{a}, {b}, 100, 180, 0, {c_int}]"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", 9000))
    sock.sendall(first_cmd.encode())
    sock.close()

    print("✅ 一次校正完成，機械手臂移動至第一次校正位置")
    time.sleep(5)
    # 2. 二次校正（拍照＋Template Match）

    # 直接從最新 frame（preview中可見）擷取
    ret2, frame2 = cap.read()
    if ret2:
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        template = cv2.imread(r"C:\Users\Gillion-BennyWinNB\Desktop\2025HIWIN_poolball\second_data\0507gay.png", cv2.IMREAD_GRAYSCALE)
        w, h = template.shape[::-1]
        result = cv2.matchTemplate(binary, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        cvx = max_loc[0] + w // 2
        cvy = max_loc[1] + h // 2
        print(f"🎯 二次校正中心座標：({cvx}, {cvy})")
    else:
        raise RuntimeError("⚠️ 二次校正拍照失敗，無法取得 frame2")

    # 呼叫二次校正函式
    x2, y2, _, _, _, c2_int = second_calibration_new(a, b, cvx, cvy, cue_angle)
    second_cmd = f"[{x2}, {y2}, 100, 180, 0, {c2_int}]"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", 9000))
    sock.sendall(second_cmd.encode())
    sock.close()
    time.sleep(1)
    print("✅ 二次校正完成，機械手臂移動至精修位置")


    third_result = f"[{x2}, {y2}, 80, 180, 0, {c2_int}]"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", 9000))
    sock.sendall(third_result.encode())
    sock.close()
    time.sleep(1)
    print("✅ 三次校正完成，機械手臂移動至打擊位置")
    # 3. 擊球（Arduino 控制）
    ser = serial.Serial("COM6", 9600, timeout=1)
    time.sleep(2)
    ser.write(b'40')#調整電壓!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ser.close()
    print("✅ 發送擊球指令至 Arduino")
    time.sleep(3)
    # 4. 回 Home
    move_arm_home()

#------- ⚡ 主code⚡ -------
if __name__ == "__main__":
    try:
    
        # 這裡可以放置其他需要持續運行的程式碼
        # 例如：接收來自機械手臂的回傳訊息等
        time.sleep(0.1)  # 避免 CPU 使用率過高
        MAIN()
        
    except KeyboardInterrupt:
        pass
    
    stop_preview.set()
    time.sleep(0.1)
    cap.release()
    cv2.destroyAllWindows()