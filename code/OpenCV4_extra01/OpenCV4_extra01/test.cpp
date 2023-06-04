#include <iostream>
#include<opencv2\opencv.hpp>
#include "opencv2/xfeatures2d.hpp"


using namespace cv;
using namespace std;
using namespace xfeatures2d;
int main()
{
	Mat matSrc = imread("E:/OpenCV_intro/test_img/01.jpg");
	Mat draw;


	std::vector<KeyPoint> keypoints;
	auto sift_detector = SIFT::create();
	sift_detector->detect(matSrc, keypoints);


	drawKeypoints(matSrc, keypoints, matSrc);
	imshow("gray", matSrc);
	waitKey(0);
	return 0;
}