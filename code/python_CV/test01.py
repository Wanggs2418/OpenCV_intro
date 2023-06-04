# use pytorch38 env
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from chapter6 import stackImages

def getContours():
    


img = cv.imread("img/03.jpg")
img_blur1 = cv.GaussianBlur(img, (7,7), 1)
img_blur5 = cv.GaussianBlur(img, (7,7), 5)
img_canny = cv.Canny(img, 50, 50)
img_canny_ref = cv.Canny(img, 100, 100)
# 创建空白图片-全黑型
img_black = np.ones_like(img)
# img_white = np.zeros_like(img)


stack_img = stackImages(1.2, ([img, img_blur1, img_blur5], [img_canny, img_canny_ref, img_black]))

cv.imshow("all_image", stack_img)
cv.waitKey(0)