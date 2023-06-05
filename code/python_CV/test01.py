# use pytorch38 env
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from chapter6 import stackImages

def getContours(img_gray):
    contours, hierarchy = cv.findContours(img_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        # print(area)
        if area > 50:
            cv.drawContours(img, cnt, -1, (255,0,0), 1)
            # 轮廓弧长，参数2：受否是封闭的或仅仅是一个曲线
            peri = cv.arcLength(cnt, True)
            # 估计轮廓形状
            approx = cv.approxPolyDP(cnt, 0.02*peri, True)
            # print(approx)
            objCor = len(approx)
            # print(objCor)
            x, y, w, h = cv.boundingRect(approx)
            if objCor == 3:
                objectType = "Tri"
            elif objCor == 4:
                # 长宽比判定
                aspRatio = w / float(h)
                if aspRatio > 0.98 and aspRatio < 1.03:
                    objectType = "Square"
                else:
                    objectType = "Circles"
            elif objCor > 4:
                objectType = "Circles"
            else:
                objectType = "None"

            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            # cv.putText(img, objectType,
            #             (x+(w//2)-10,y+(h//2)-10),cv.FONT_HERSHEY_COMPLEX,0.7,
            #             (0,0,0),2)

            

img = cv.imread("img/03.jpg")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# img_Contour = img.copy()

img_blur1 = cv.GaussianBlur(img, (7,7), 1)
img_blur5 = cv.GaussianBlur(img, (7,7), 5)
img_canny = cv.Canny(img, 50, 50)
img_canny_ref = cv.Canny(img, 100, 100)

# ret, img_threshold = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY)
ret, img_threshold = cv.threshold(img_gray, 150, 255, 0)
# 创建空白图片-全黑型
# img_black = np.ones_like(img)
# img_white = np.zeros_like(img)

# getContours(img_threshold)
getContours(img_canny)

stack_img = stackImages(1.2, ([img, img_blur1, img_blur5], [img_gray, img_canny, img_threshold]))

cv.imshow("all_image", stack_img)
cv.waitKey(0)