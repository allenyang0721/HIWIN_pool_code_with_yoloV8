
# 2025HIWIN_poolball 撞球決策系統

本專案為參與 2025 HIWIN 機械手臂競賽所開發的「撞球擊球策略判斷系統」，結合 RealSense 相機與 YOLOv8 模型辨識球桌球體位置，進行最佳擊球路徑與策略分析，並導出分析結果。

## 📦 專案結構

```
2025HIWIN_poolball/
├── src/                    # 主程式與邏輯
├── config/                 # 設定檔（如 poolball.yaml）
├── yoloV8_train/           # YOLO 模型與訓練資料（.pt 不應推送 Git）
├── output/                 # 拍照結果與分析圖片、JSON 檔
├── data_sample/            # 小型圖片測試資料
├── models/                 # 模型說明與連結
├── requirements.txt        # Python 套件需求
└── README.md               # 本說明文件
```

## 🚀 執行方式

```bash
python src/poolball_main.py
```

1. 啟動後會開啟攝影機，按下「空白鍵」拍照。
2. 使用指定的 YOLOv8 模型進行球類辨識。
3. 自動計算直擊或反彈的最佳路徑。
4. 結果將顯示於畫面，並輸出至 `output/`：

   - `pool_analysis.png`：圖像標註路徑
   - `route_data.json`：路徑資料（含角度與球座標）

## 🧠 使用模型

請將你訓練的最佳 YOLOv8 權重（`.pt`）放於 `yoloV8_train/` 資料夾中，並於程式中設定正確檔名。

## 📋 requirements.txt 建議內容

```txt
ultralytics
numpy
opencv-python
```

可使用 pip 安裝：

```bash
pip install -r requirements.txt
```

## ⚠️ 注意事項

- `.pt`, `.zip`, `.png`, `output/` 等資料皆應透過 `.gitignore` 排除，避免污染 Git 紀錄。
- 建議大型檔案可改以 Release 發佈。
- 若於 Colab 執行請確保有 camera 模擬資料或改為讀圖模式。

## 📮 聯絡方式

本專案由 allen 製作，若有合作需求請私訊。

