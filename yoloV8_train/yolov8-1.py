from ultralytics import YOLO

# 類別對應名稱
class_names = {
    0: "cue_ball",    # 母球
    1: "ball_1",
    2: "ball_2",
    3: "ball_3",
    4: "pocket"       # 洞口
}

# 載入模型
model = YOLO(r"C:\Users\Gillion-BennyWinNB\Desktop\2025HIWIN_poolball\yoloV8_train\956PT_0428_best.pt")
results = model(r"C:\Users\Gillion-BennyWinNB\Desktop\2025HIWIN_poolball\take_images\photo_000.png")

# 初始化資料結構
detected_objects = {
    "cue_ball": None,
    "ball_1": None,
    "ball_2": None,
    "ball_3": None,
    "pockets": []  # 可能有多個洞口
}

# 處理每個辨識結果
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()
        name = class_names.get(cls_id)

        if name == "pocket":
            detected_objects["pockets"].append({"x": cx, "y": cy, "confidence": conf})
        else:
            detected_objects[name] = {"x": cx, "y": cy, "confidence": conf}

# 顯示結果
import pprint
pprint.pprint(detected_objects)
