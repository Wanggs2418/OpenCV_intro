//OpenCV����Ҫ��ͷ�ļ���include ��opencv2/opencv.hpp�������Ե���ÿ�� Opencv ģ���ͷ�ļ�
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;
int main()
{
	// Matrix ����
	Mat img = imread("img//0100.jpg", IMREAD_GRAYSCALE); //�Ҷ�ͼ��IMREAD_GRAYSCALE
	if (img.empty()) {
		printf("could not load image!\n");
		return -1;
	}
	namedWindow("window01", WINDOW_FREERATIO);
	imshow("window01", img); //imshow�޷�����ͼƬAUTOSIZE
	waitKey(0); // 1 ��ʾ1ms
	destroyAllWindows();
	return 0;
}
