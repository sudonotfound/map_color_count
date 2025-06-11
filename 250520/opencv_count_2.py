import cv2
import numpy as np
import os

# 入力画像ファイル（Windows形式のパスに修正）
# 現在のフォルダ内または適切なパスを指定してください
image_path = 'C:\\Users\\USER\\Desktop\\4color\\250520\\0521_test.png'  # Windowsパス形式

# 画像が存在するか確認
if not os.path.exists(image_path):
    print(f"画像ファイルが見つかりません: {image_path}")
    print("現在の作業ディレクトリ:", os.getcwd())
    exit()

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

# 面積閾値（例えば100ピクセル未満は除外）
area_threshold = 100

# 大きい輪郭だけ抽出
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= area_threshold]

# 領域の個数を出力
print(f"RGB(0,0,255) に近い色の大きな領域数: {len(filtered_contours)}")

# 元画像に枠を描画
output_image = image.copy()
cv2.drawContours(output_image, filtered_contours, -1, (0, 255, 255), 2)

# 画像の保存
cv2.imwrite('output_contours_2.png', output_image)
cv2.imwrite('output_mask_2.png', mask)
print("出力画像を保存しました：output_contours_2.png, output_mask_2.png")

# 結果を表示
cv2.imshow('Original + Contours', output_image)
cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()