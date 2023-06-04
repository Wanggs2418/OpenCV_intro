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
		cout << "摄像头调用失败!";
		return -1;
	}
	
	video.set(CAP_PROP_FPS, 30);//设置帧率
	cout << "视频帧率" << video.get(CAP_PROP_FPS) << endl;
	cout << "视频宽度" << video.get(CAP_PROP_FRAME_WIDTH) << endl;
	while (1)
	{
		Mat frame;
		video >> frame;
		if (frame.empty())
		{
			break;
		}
		imshow("处理前的视频", frame); //显示每一帧
		//cvtColor(frame, edges, COLOR_BGR2RGB);//彩色图片转换为灰色
		cvtColor(frame, edges, COLOR_BGR2GRAY);//彩色图片转换为灰色
		blur(edges, edges, Size(7, 7)); //模糊操作
		Canny(edges, edges, 0, 30, 3);//边缘检测
		//blur(frame, frame, Size(7, 7)); //模糊操作
		//Canny(frame, frame, 0, 30, 3);//边缘检测

		imshow("处理后的视频", edges); //显示每一帧
		////每秒放多少帧，计算每个帧的时间间隔
		uchar c = waitKey(1000 / video.get(CAP_PROP_FPS));
		if (c == 'q')
		{
			break;
		}
	}



	return 0;
}