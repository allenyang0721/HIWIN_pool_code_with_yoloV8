import cv2
import numpy as np

# 圖片路徑
image_path = r"C:\Users\Gillion-BennyWinNB\Downloads\gg.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("❌ 圖片讀取失敗，請確認路徑是否正確")
else:
    # 二值化
    _, binary = cv2.threshold(img, 118, 255, cv2.THRESH_BINARY)

    # 建立 kernel（可調整大小與形狀）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # 膨脹
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # 侵蝕
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # 儲存結果
    output_path = r"C:\Users\Gillion-BennyWinNB\Desktop\2025HIWIN_poolball\captured_binary\binarized_output.png"
    cv2.imwrite(output_path, binary)

    # 顯示前後對比
    cv2.imshow("Original Binary", binary)
    cv2.imshow("After Dilation + Erosion", eroded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
