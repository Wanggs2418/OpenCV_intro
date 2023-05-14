# OpenCV_intro

## 1.基础

### 1.1 图像读取和显示

直接使用 `imshow` 时，默认为 `AUTOSIZE`，需要拖动改变窗口大小，需设置新的窗口`namedWindow("示例图像1", WINDOW_FREERATIO);` 设置 `WINDOW_FREERATIO` 属性。

```c++
//OpenCV中主要的头文件是include “opencv2/opencv.hpp”它可以调用每个 Opencv 模块的头文件
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;
int main()
{
	// Matrix 矩阵
	//本地文件路径，E:/OpenCV_intro/test_img/01.jpg
	Mat img = imread("E:/OpenCV_intro/test_img/01.jpg", IMREAD_GRAYSCALE); //灰度图像IMREAD_GRAYSCALE
	if (img.empty()) {
		printf("could not load image!\n");
		return -1;
	}
	namedWindow("示例图像1", WINDOW_FREERATIO);//FREEATIO
	imshow("示例图像1", img); //imshow无法调整图片,默认AUTOSIZE
	waitKey(0); // 1 表示1ms
	destroyAllWindows();
	return 0;
}

```

### 1.2 色彩转换

- cvtColor: 色彩转换空间
- imwrite: 图像保存





















