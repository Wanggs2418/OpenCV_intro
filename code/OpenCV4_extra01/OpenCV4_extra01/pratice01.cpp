#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
	Mat img;
	Mat edges;
	VideoCapture video;
	video.open(0);
	if (!video.isOpened())
	{
		cout << "����ͷ����ʧ��!";
		return -1;
	}
	
	video.set(CAP_PROP_FPS, 30);//����֡��
	cout << "��Ƶ֡��" << video.get(CAP_PROP_FPS) << endl;
	cout << "��Ƶ���" << video.get(CAP_PROP_FRAME_WIDTH) << endl;
	while (1)
	{
		Mat frame;
		video >> frame;
		if (frame.empty())
		{
			break;
		}
		imshow("����ǰ����Ƶ", frame); //��ʾÿһ֡
		//cvtColor(frame, edges, COLOR_BGR2RGB);//��ɫͼƬת��Ϊ��ɫ
		cvtColor(frame, edges, COLOR_BGR2GRAY);//��ɫͼƬת��Ϊ��ɫ
		blur(edges, edges, Size(7, 7)); //ģ������
		Canny(edges, edges, 0, 30, 3);//��Ե���
		//blur(frame, frame, Size(7, 7)); //ģ������
		//Canny(frame, frame, 0, 30, 3);//��Ե���

		imshow("��������Ƶ", edges); //��ʾÿһ֡
		////ÿ��Ŷ���֡������ÿ��֡��ʱ����
		uchar c = waitKey(1000 / video.get(CAP_PROP_FPS));
		if (c == 'q')
		{
			break;
		}
	}



	return 0;
}