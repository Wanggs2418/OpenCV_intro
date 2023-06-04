import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def empty(a):
    pass

def imgBGR2RGB(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


img = cv.imread("img/03.jpg")
img_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

cv.namedWindow("example_windows")
cv.resizeWindow("example_windows", 640, 340)
cv.createTrackbar("hue_min", "example_windows", 0, 179, empty)
# cv.createTrackbar("hue_max", "example_windows", 179, 179, empty)
cv.createTrackbar("hue_max", "example_windows", 176, 179, empty)
cv.createTrackbar("sat_min", "example_windows", 0, 255, empty)
# cv.createTrackbar("sat_max", "example_windows", 255, 255, empty)
cv.createTrackbar("sat_max", "example_windows", 220, 255, empty)
# cv.createTrackbar("val_min", "example_windows", 0, 255, empty)
cv.createTrackbar("val_min", "example_windows", 89, 255, empty)
cv.createTrackbar("val_max", "example_windows", 255, 255, empty)


while(1):
    h_min = cv.getTrackbarPos("hue_min", "example_windows")
    s_min = cv.getTrackbarPos("sat_min", "example_windows")
    v_min = cv.getTrackbarPos("val_min", "example_windows")
    h_max = cv.getTrackbarPos("hue_max", "example_windows")
    s_max = cv.getTrackbarPos("sat_max", "example_windows")
    v_max = cv.getTrackbarPos("val_max", "example_windows")
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower_array = np.array([h_min, s_min, v_min])
    upper_array = np.array([h_max, s_max, v_max])
    mask = cv.inRange(img_HSV, lower_array, upper_array)
    result_img = cv.bitwise_and(img, img, mask)

    # plt.subplot(131), plt.imshow(imgBGR2RGB(img_HSV)), plt.title("HSV")
    # plt.subplot(132), plt.imshow(imgBGR2RGB(mask)), plt.title("mask")
    # plt.subplot(133), plt.imshow(imgBGR2RGB(result_img)), plt.title("result_img")
    # plt.show()

    # cv.imshow("HSV", img_HSV)
    # cv.imshow("mask", mask)
    # cv.imshow("mask_img", result_img)

    # 堆叠照片实现类似 subplot 的功能
    stack_img = stackImages(1.2, ([img_HSV, img], [mask, result_img]))
    cv.imshow("hstack_img", stack_img)

    k = cv.waitKey(1) # 注意使用 1，而非0,可以实时改变
    if k == "q":
        break
cv.destroyAllWindows()
