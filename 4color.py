import cv2
import numpy as np

# print("OpenCV version")
# print(cv2.__version__)

#画像読み込み関数
image=cv2.imread("4color_map.png")

#画像表示関数
cv2.imshow("CarrotCake",image)
#cv2.waitKey(0)


# bgrでの色抽出
# # オレンジ色のBGR範囲を少し広げる
bgrLower = np.array([180, 100, 0])  # 抽出する色の下限（オレンジに近い下限）
bgrUpper = np.array([255, 180, 50])  # 抽出する色の上限（オレンジに近い上限）

img_mask = cv2.inRange(image, bgrLower, bgrUpper) # bgrからマスクを作成
extract = cv2.bitwise_and(image, image, mask=img_mask) # 元画像とマスクを合成
cv2.imwrite('extract.png',extract)
#cv2.imshow('extract.png',extract)
cv2.imshow('extract.png',img_mask)
cv2.waitKey(0) 