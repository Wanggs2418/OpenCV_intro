//OpenCV中主要的头文件是include “opencv2/opencv.hpp”它可以调用每个 Opencv 模块的头文件
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;
int main()
{
	// Matrix 矩阵
	Mat img = imread("img//0100.jpg", IMREAD_GRAYSCALE); //灰度图像IMREAD_GRAYSCALE
	if (img.empty()) {
		printf("could not load image!\n");
		return -1;
	}
	namedWindow("window01", WINDOW_FREERATIO);
	imshow("window01", img); //imshow无法调整图片AUTOSIZE
	waitKey(0); // 1 表示1ms
	destroyAllWindows();
	return 0;
}
