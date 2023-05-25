import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


kernel = np.ones((5, 5), np.uint8)
img = cv.imread("img/03.jpg")

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_blur = cv.GaussianBlur(img, (7, 7), 0)
img_canny = cv.Canny(img, 150, 200) # 边缘检测
img_dilation = cv.dilate(img_gray, kernel, iterations=1) # 迭代一次
img_erode = cv.erode(img_gray, kernel, iterations=1) # 迭代一次

# cv.imshow("name01", img_gray)
# cv.imshow("blur", img_blur)
cv.imshow("canny", img_canny)
cv.imshow("canny_dilation", img_dilation)
cv.imshow("canny_erode", img_erode)


c = cv.waitKey(0)

