import cv2
import numpy as np

# 画像ファイルのパスを指定
image_path = "test1.png"

# 画像を読み込み (RGB -> BGR)
image = cv2.imread(image_path)

# 色の定義 (RGB形式 -> BGR形式)
colors = {
    "glen": (80, 176, 0),         # (Blue, Green, Red)
    "yellow": (0, 255, 255),
    "porple": (160, 48, 112),
    "red": (0, 0, 255),
    "Light Green": (0, 255, 145)
}

# 各色の領域をカウントする
region_counts = {}

for color_name, bgr_value in colors.items():
    # 指定された色のマスクを作成
    mask = cv2.inRange(image, bgr_value, bgr_value)
    
    # 輪郭を検出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 領域の数を記録
    region_counts[color_name] = len(contours)

# 結果を表示
for color_name, count in region_counts.items():
    print(f"{color_name}: {count} regions")
