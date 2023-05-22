#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
	Mat img = imread("E:/AI_learning/06CV_intro/OpenCV_intro/test_img/bg01.jpg");;
	Mat img32;
	img.convertTo(img32, CV_32F, 1 / 255.0, 0); //0-255转换为0-1
	
	//Mat imgs[3]; //定义的分离向量数组
	vector<Mat> imgs;
	split(img, imgs);
	Mat img00 = imgs[0];
	Mat img01 = imgs[1];
	Mat img02 = imgs[2];

	return 0;
}