import os

# 設定要處理的資料夾路徑
folder_path = r'C:\Users\Gillion-BennyWinNB\Desktop\labels_hilight'  # 修改成你的資料夾路徑

# 遍歷資料夾中所有的 .txt 檔案
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)

        # 讀取檔案前10行
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            lines = lines[:10]  # 只取前10行

        # 寫回檔案（覆蓋原內容）
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)

        print(f"已處理檔案：{filename}")
