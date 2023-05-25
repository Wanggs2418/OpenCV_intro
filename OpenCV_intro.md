# OpenCV_intro

# 第一部分(C++版)

>  [官网教程](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)

## 0.C++

### 0.1 vector

[vector 浅析](https://www.runoob.com/w3cnote/cpp-vector-container-analysis.html)

`std::vector`

- 存放任意类型的动态数组

**基本操作**

- 增加元素

  ```c++
  void push_back(const T& x):向量尾部增加一个元素X
  iterator insert(iterator it,int n,const T& x):向量中迭代器指向元素前增加n个相同的元素x
  ```

- 删除元素

  ```c++
  void pop_back():删除向量中最后一个元素
  void clear():清空向量中所有元素
  ```

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
    imwrite("E:/OpenCV_intro/test_img/01_gray.jpg", img); //保存
	destroyAllWindows();
	return 0;
}

```

### 1.2 运行源码中的示例

位置：`D:\opencv\sources\samples\cpp`

**1.边缘检测**

点击 `源文件`，添加 `现有项目` 中的 `edge.cpp`，提示找不到对应的 lib 文件的话可以重启系统。

使用 `CMD`在生成的 `.exe` 文件夹下对目标照片操作：

```cmd
.\OpenCV4_extra01 01.jpg
```

![](img/21.jpg)

**2.K聚类**

`kmeans.cpp`：直接构建并运行即可，不需要参数

按空格键可交互，`esc` 键退出

**3.借助相机进行目标跟踪**

`camshiftdemo.cpp`: 调用相机进行目标跟踪

命令参数的输入可在 `debug|×64` 右键，在调试选项中输入：

![](img/22.jpg)

按键说明：

```c++
string hot_keys =
    "\n\nHot keys: \n"
    "\tESC - quit the program\n"
    "\tc - stop the tracking\n"
    "\tb - switch to/from backprojection view\n"
    "\th - show/hide object histogram\n"
    "\tp - pause video\n"
    "To initialize tracking, select the object with mouse\n";
```

## 2. Mat 容器

[官网 Mat 类解读](https://docs.opencv.org/4.6.0/d3/d63/classcv_1_1Mat.html#a2ec3402f7d165ca34c7fd6e8498a62ca)

Mat 类： OpenCV 用于存储矩阵数据类型，类似于 int，double 类型

Mat 能存储的数据：

![](img/25.jpg)

![](img/23.jpg)

### 2.1 创建 Mat 类

- 利用矩阵的宽，高，类型参数创建 Mat 类
- 利用 Size() 结构和数据类型参数创建 Mat 类
- 利用原有的 Mat 类创建

```c++
cv::Mat::Mat(	
int 	rows,
int 	cols,
int 	type 
)	
//CV_8U(n) 其中 n 用来构建多通道数，最大为 512
//CV_8UC1,CV_64FC1 等是从 1-4
Mat demo(3, 3, CV_8U)
Mat a(Size(3,3), CV_8U)

//利用已有的 Mat 类创建
//[2,5)第2行到第5行，第2列到第5列
c = Mat(a, Range(2,5), Range(2,5))
```

**赋值 Mat 类**

```c++
Mat demo(3, 3, CV_8U, Scalar(0))
//eye,diag,zeros,ones
```

### 2.2 读取 Mat 类

```c++
a.cols
a.rows
a.step
```

![](img/26.jpg)

**at 方法读取**

**需要知道读取的数据类型**

Vec3b: 3 个通道 uchar 类型的

Vec4i: 4 个通道 int 类型的

Vec2d: 2 个通道 double 类型的

```c++
//单通道
int value = (int)a.at<uchar>(0,0);
//多通道，Vec3b 3通道，中的 b 代表uchar类型，d 代表double
cv::Vec3b vc3 = b.at<cv::Vec3b>(0, 0);
int first = (int)vc3.val[0];
```

**地址定位方式访问**

```c++
//不用考虑数据类型
//单通道(row, col, channel)
(int)(*(b.data + b.step[0] * row + b.step[1] * col + channel));
```

### 2.3 符号运算

数据类型和尺寸的一致性

矩阵乘积：

- 矩阵的乘积
- 内积，数据个数一致即可
- 对应元素相乘

```c++
a*b;
a.dot(b);
a.mul(b);
```



![](img/27.jpg)

```c++
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
	system("color F0");
	//CV_8U类型，通道数1
	Mat a(3, 3, CV_8UC1);
	Mat b(Size(4, 4), CV_8UC1);

	//赋值
	//5×5×3
	Mat c3(5, 5, CV_8UC3, Scalar(4, 5, 6));
	Vec3b vc3_uchar = c3.at<Vec3b>(0, 0);
	cout << vc3_uchar << endl;
	cout <<"3通道赋值\n" << c3 << endl;
	cout << "step[0]:" << c3.step[0] << endl;
	cout << "step[1]:" << c3.step[1] << endl;
	//枚举赋值
	Mat d = ( cv::Mat_<int>(1, 5) << 1, 2, 3, 4, 5 );
	//产生对角矩阵

	Mat e = Mat::diag(d);
	//从0开始计数
	Mat f = Mat(e, Range(2, 4), Range(2, 4));
	cout << "枚举赋值\n" << d << endl;
	cout << "对角阵选取\n" << e << endl;
	cout << "选取\n" <<f << endl;

}
```

### 2.4 图像读取，显示保存

**imread**

[官方说明](https://docs.opencv.org/4.6.0/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56)

imread 读入，imshow 显示

```c++
// a.imread("路径"，flags窗口属性的标志)
```

**namedWindow**

```c++
//name 窗口名
nameWindow(name, flags);
imshow(winname, mat);
//filename 保存的路径，Mat 类图像的名字,图片属性的设置 
imwrite(filename, img, params)
```

**Image Watch**

[ImageWatch](https://marketplace.visualstudio.com/search?term=image&target=VS&category=Tools&vsVersion=&subCategory=All&sortBy=Relevance) | [Image Watch VS 2017 版](https://marketplace.visualstudio.com/items?itemName=VisualCPPTeam.ImageWatch2017)

![](img/28.jpg)

### 2.5 视频加载和调用摄像头

VideoCapture

filename: 读取的视频名称

apiPreference: 读取数据时设置的属性，如编码格式，是否调用 OpenNI

```c++
//VideoCapture(filename, apiPreference);
//对每帧视频的读取
int main() {
	VideoCapture video;
	video.open("no.mp4");
	if (!video.isOpened())
	{
		cout << "视频路径错误或者不存咋!";
		return -1;
	}
	//property
	cout <<"视频帧率" << video.get(CAP_PROP_FPS) << endl;
	cout << "视频宽度" << video.get(CAP_PROP_FRAME_WIDTH) << endl;
	while (1)
	{
		Mat frame;!
		video >> frame;//存储每一帧的图像
		if (frame.empty())
		{
			break;
		}
		imshow("video_name", frame); //显示每一帧
		////每秒放多少帧，计算每个帧的时间间隔
		uchar c = waitKey(1000/ video.get(CAP_PROP_FPS)); 
		if (c == 'q')
		{
			break;
		}
	}
	return 0;
}
```

**调用摄像头**

```c++
int main() {
	Mat img;
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
		imshow("video_name", frame); //显示每一帧
		////每秒放多少帧，计算每个帧的时间间隔
		uchar c = waitKey(1000 / video.get(CAP_PROP_FPS));
		if (c == 'q')
		{
			break;
		}
	}

	video >> img;
	if (img.empty())
	{
		cout << "获取图像失败" << endl;
		return -1;
	}
	//是否为彩色图片,3通道彩色图片
	bool isColor = (img.type() == CV_8UC3);

	VideoWriter writer;
	int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');//选择编码格式
	double fps = 25.0; //设置帧率

	string filename = "E:/Desktop/test01.avi";
	writer.open(filename, codec, fps, img.size(), isColor);
	if (!writer.isOpened())
	{
		cout << "打开视频文件失败" << endl;
	}


	while (1)
	{
		if (!video.read(img))
		{
			cout << "摄像头断开链接或视频读取完成" << endl;
			return -1;
		}
		writer.write(img);//图像写入视频流，writer << img
		imshow("显示写入视频的图像", img);
		char c = waitKey(50);
		if (c == 'q')
		{
			break;
		}
		//video.release();
		//writer.release();
	}
	return 0;
}
```







### 2.6 颜色转换

**RGB**

8U：0-255；f32：0-1；d64：0-1

```c++
void cv::Mat::convertTo	(	OutputArray 	m,
int 	rtype,
double 	alpha = 1,
double 	beta = 0 
)		constc
//0-225转换为0-1
a.convertTo(b, CV_32F, 1/255.0, 0)
```

m：输出图像；rtype：转换后的数据类型；alpha：缩放系数；beta：平移系数

**HSV**

H: Hue 色度，即颜色；S: Saturation，饱和度，深浅；V: value，亮度

**GRAY** 

0-127-255

只能由彩色转换为灰色，反之则不行。

```c++
void cv::cvtColor	(	InputArray 	src,
OutputArray 	dst,
int 	code,
int 	dstCn = 0 
)		
```

src: 原始图像

dst: 转换后的图像

code: 颜色转换的标志

dstCn: 目标图像中的通道数，如果参数为 0，则从自动导出

```c++
int main() {
	Mat img = imread("E:/AI_learning/06CV_intro/OpenCV_intro/test_img/bg01.jpg");;
	Mat img32;
	Mat HSV, HSV32;
	img.convertTo(img32, CV_32F, 1 / 255.0, 0); //0-255转换为0-1
	cvtColor(img, HSV, COLOR_BGR2HSV, 0);
	cvtColor(img32, HSV32, COLOR_BGR2HSV, 0);

	Mat gray0, gray1;
	cvtColor(img, gray0, COLOR_BGR2GRAY, 0); //转换为灰度,BRG比RGB的排列顺序不同，结果更亮
	cvtColor(img, gray1, COLOR_RGB2GRAY, 0);
	
	return 0;
}
```

### 2.7 多通道分离

[split](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga8027f9deee1e42716be8039e5863fbd9)

```c++
void cv::split	(	InputArray 	m,
OutputArrayOfArrays 	mv 
)	
```

需分离的图像；分离后的单通道图像，表现为向量类型

```c++
int main() {
	Mat img = imread("E:/AI_learning/06CV_intro/OpenCV_intro/test_img/bg01.jpg");;
	Mat img32;
	img.convertTo(img32, CV_32F, 1 / 255.0, 0); //0-255转换为0-1
	
    Mat img_merge;
	Mat imgs[3]; //定义的分离向量数组
    
	split(img, imgs);
	Mat img0 = imgs[0];
	Mat img1 = imgs[1];
	Mat img2 = imgs[2];
    
    merge(imgs, 3, img_merge);

	return 0;
}
```

[merge](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga61f2f2bde4a0a0154b2333ea504fab1d)

```c++
void cv::merge	(	const Mat * 	mv,
size_t 	count,
OutputArray 	dst 
)	

void cv::merge	(	InputArrayOfArrays 	mv,
OutputArray 	dst 
)	
```

# 第二部分(Python版)

> [opencv-python 原版文档](https://docs.opencv.org/4.1.2/d6/d00/tutorial_py_root.html)
>
> [opencv-python 中文官方文档](http://www.woshicver.com/)

## 1.简介

**直接安装**

[**Install OpenCV-Python in Windows**](https://docs.opencv.org/4.1.2/d5/de5/tutorial_py_setup_in_windows.html)

```bash
pip install numpy
pip install matplotlib
pip install opencv-python
```

**包导入**

```python
import cv2 as cv
print( cv.__version__ )
```

### 1.1图像

[**Getting Started with Images**](https://docs.opencv.org/4.1.2/dc/d2e/tutorial_py_image_display.html)

**[cv.imread()](https://docs.opencv.org/4.1.2/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56)**: 读取图片，("01.jpg", 0)，0 表示读取方式为灰度

**[cv.imshow()](https://docs.opencv.org/4.1.2/d7/dfc/group__highgui.html#ga453d42fe4cb60e5723281a89973ee563)**: 显示

**[cv.imwrite()](https://docs.opencv.org/4.1.2/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce)**: 保存

彩色图片在 OpenCV 默认以BGR 模式读取，在 Matplotlib 中则以 RGB 模式显示

### 1.2视频

[**Getting Started with Videos**](https://docs.opencv.org/4.1.2/dd/d43/tutorial_py_video_display.html)

**[cv.VideoCapture()](https://docs.opencv.org/4.1.2/d8/dfe/classcv_1_1VideoCapture.html)**:  per second of frames，fps

**[cv.VideoWriter()](https://docs.opencv.org/4.1.2/dd/d9e/classcv_1_1VideoWriter.html)**: 

VGA: 640×480

HD: 1280×720

FHD: 1920×1080

4K: 3840×2160

灰度图像：0-255，除了 0 表示黑，255 表示白，中间 244 个值表示灰度；

关于 propId 数值对应的含义见：**[Flags for video I/O](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d)**

```python
framewidth = 640
frameheight = 480
fps = 30 # 设置帧率

# 其中propId是0到18之间的一个数字。每个数字表示视频的属性
# cap.set(propID, value)

# cap = cv2.VideoCapture("video/test01.avi")
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, framewidth) #id-3
cap.set(cv.CAP_PROP_FRAME_HEIGHT, frameheight) #id-4
cap.set(cv.CAP_PROP_FPS, fps) #id-5
cap.set(cv.CAP_PROP_BRIGHTNESS, 1) #id-10

while cap.isOpened():
    success, img = cap.read()
    cv.imshow("video", img)
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow("gray", gray)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
```

**借助 VideoWriter 对象保存**

- 文件路径，包含文件名 filename

- 设置编码形式 FourCC, 4-bytes, 如  MJPG (.mp4), DIVX (.avi)

  `cv.VideoWriter_fourcc('M','J','P','G')` 或者 `(*'MJPG')`

- frames per second (fps)

- frame size 

- isColor, 为真则以彩色图片保存；否则以灰度图片保存

**cap 对象，实时获取摄像机视频帧 => frame => out 对象**

```python
out = cv.VideoWriter('output.avi', fourcc, 20.0, (640,  480)) #定义一个VideoWriter对象，用于接受视频帧

out.write(frame) #将获取的帧写入对象中
cv.imshow('frame', frame)# 显示获取的帧
```

### 1.3 绘制图像

[**Drawing Functions in OpenCV**](https://docs.opencv.org/4.1.2/dc/da5/tutorial_py_drawing_functions.html)

### 1.4 鼠标刷

[**Mouse as a Paint-Brush**](https://docs.opencv.org/4.1.2/db/d5b/tutorial_py_mouse_handling.html)

### 1.5 色彩棒

[**Trackbar as the Color Palette**](https://docs.opencv.org/4.1.2/d9/dc8/tutorial_py_trackbar.html)

## 2.图片操作

[**Image Processing in OpenCV**](https://docs.opencv.org/4.1.2/d2/d96/tutorial_py_table_of_contents_imgproc.html)

图片的坐标轴表示，y 轴向下为正：

![](img/30.jpg)

### 2.1 色彩变换

[**Changing Colorspaces**](https://docs.opencv.org/4.1.2/df/d9d/tutorial_py_colorspaces.html)

将图片从一种色彩形式转换 (convert, cvt) 为另一种，常见的如： **BGR ↔ Gray, BGR ↔ HSV**

```python
 flags = [i for i in dir(cv) if i.startswith('COLOR_')]
```

### 2.2 形态变换

[**Morphological Transformations**](https://docs.opencv.org/4.1.2/d9/d61/tutorial_py_morphological_ops.html)

适用于二值图 ( binary images)，借助矩阵操作 (使用 numpy 包)。

 **[cv.erode()](https://docs.opencv.org/4.1.2/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb)**: 腐蚀

**[cv.dilate()](https://docs.opencv.org/4.1.2/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c)**: 膨胀

![](img/29.jpg)

```python
import cv2 as cv
import numpy as np

kernel = np.ones((5, 5), np.uint8)
img = cv.imread("img/03.jpg")

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_blur = cv.GaussianBlur(img, (7, 7), 0)
img_canny = cv.Canny(img, 150, 200) # 边缘检测
img_dilation = cv.dilate(img_gray, kernel, iterations=1) # 迭代一次
img_erode = cv.erode(img_gray, kernel, iterations=1) # 迭代一次

# cv.imshow("name01", img_gray)
# cv.imshow("blur", img_blur)
cv.imshow("canny", img_canny)
cv.imshow("canny_dilation", img_dilation)
cv.imshow("canny_erode", img_erode)

c = cv.waitKey(0)
```









