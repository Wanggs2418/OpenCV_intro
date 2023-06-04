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