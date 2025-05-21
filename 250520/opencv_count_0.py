import cv2
import numpy as np

# 入力画像ファイル
image_path = '/home/uehara/map_color_count/250520/0521_test.png'

# 画像を読み込む（BGR）
image = cv2.imread(image_path)
if image is None:
    print("画像の読み込みに失敗しました")
    exit()

# BGR → HSV に変換
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# RGB(0, 0, 255) → BGR(255, 0, 0) → HSV ≒ (120, 255, 255)
target_hue = 120  # 青のH値
lower_bound = np.array([target_hue - 5, 200, 200])
upper_bound = np.array([target_hue + 5, 255, 255])

# 指定色の範囲でマスク作成
mask = cv2.inRange(hsv, lower_bound, upper_bound)

# 輪郭検出
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 領域の個数を出力
print(f"RGB(0,0,255) に近い色の領域数: {len(contours)}")

# 元画像に枠を描画
output_image = image.copy()
cv2.drawContours(output_image, contours, -1, (0, 255, 255), 2)

# 画像の保存
cv2.imwrite('output_contours.png', output_image)
cv2.imwrite('output_mask.png', mask)
print("出力画像を保存しました：output_contours.png, output_mask.png")

# 結果を表示
cv2.imshow('Original + Contours', output_image)
cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()