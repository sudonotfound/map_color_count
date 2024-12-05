import numpy as np
import cv2
import matplotlib.pyplot as plt

#dir = "/Users/goooo/programming/test/"
dir = "/Users/USER/Desktop/4color/"
def display(img,cmap=None):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap=cmap)

coffee_img = cv2.imread(dir + "4color_map.png")




# 中央値フィルターを使ったぼかしの適用
coffee_blur = cv2.medianBlur(coffee_img ,7)
display(coffee_blur)

# グレースケールに変換

gray_coffee = cv2.cvtColor(coffee_blur,cv2.COLOR_BGR2GRAY)
display(gray_coffee,cmap='gray')

# 2値化処理
# 画像の特徴的な部分、関心のある部分を抽出するように変換する処理

ret, coffee_thresh = cv2.threshold(gray_coffee,70,255,cv2.THRESH_BINARY_INV)
display(coffee_thresh,cmap='gray')

contours, hierarchy = cv2.findContours(coffee_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(coffee_img, contours, i, (255, 0, 0), 5)

display(coffee_img)
print(len(contours))
