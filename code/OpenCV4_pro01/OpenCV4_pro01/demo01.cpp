//OpenCV����Ҫ��ͷ�ļ���include ��opencv2/opencv.hpp�������Ե���ÿ�� Opencv ģ���ͷ�ļ�
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;
int main()
{
	// Matrix ����
	//�����ļ�·����E:/OpenCV_intro/test_img/01.jpg
	Mat img = imread("E:/OpenCV_intro/test_img/01.jpg", IMREAD_GRAYSCALE); //�Ҷ�ͼ��IMREAD_GRAYSCALE
	if (img.empty()) {
		printf("could not load image!\n");
		return -1;
	}
	namedWindow("ʾ��ͼ��1", WINDOW_FREERATIO);//FREEATIO
	imshow("ʾ��ͼ��1", img); //imshow�޷�����ͼƬ,Ĭ��AUTOSIZE
	waitKey(0); // 1 ��ʾ1ms
	destroyAllWindows();
	return 0;
}