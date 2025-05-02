import detections
# 假設已讀入 YOLO 的 .txt 格式標籤為 detections（class_id 為 0~4）
# 並已轉換為絕對座標 (x_center, y_center)

mom = None         # 母球
target = None      # 目標球
obstacles = []     # 障礙球
holes = []         # 洞口

ball_dict = {}     # 暫存球號對應的球
for det in detections:
    class_id = det['class_id']
    x_min, y_min, x_max, y_max = det['xyxy']
    cx = int((x_min + x_max) / 2)
    cy = int((y_min + y_max) / 2)
    pos = (cx, cy)

    if class_id == 0:
        mom = pos
    elif class_id == 1:
        ball_dict[1] = pos
    elif class_id == 2:
        ball_dict[2] = pos
    elif class_id == 3:
        ball_dict[3] = pos
    elif class_id == 4:
        holes.append(pos)

# 選出目標球與障礙球
if 1 in ball_dict:
    target = ball_dict[1]
    for i in [2, 3]:
        if i in ball_dict:
            obstacles.append(ball_dict[i])
elif 2 in ball_dict:
    target = ball_dict[2]
    if 3 in ball_dict:
        obstacles.append(ball_dict[3])
elif 3 in ball_dict:
    target = ball_dict[3]

# 顯示結果
print("🎱 母球位置:", mom)
print("🎯 目標球:", target)
print("🚧 障礙球:", obstacles)
print("🕳️ 洞口:", holes)
