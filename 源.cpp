
#include<opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
using namespace cv;
using namespace std;
bool IsRight(Mat image, int x, int y, int width, int height)
{
	int i, j, k = 0;
	int index = 0;
	for (i = -1; i <= 1; i++)
	{
		if (x + i < width && x + i >= 0)
		{
			for (j = -1; j <= 1; j++)
			{
				if (y + j < height && y + j >= 0)
				{
					index = x * width + y;
					if (image.data[index] != 255)
					{
						k++;
					}
				}
			}
		}
	}
	if (k > 2)
		return true;
	else
		return false;
}
int main()
{
	
	vector<int>grayv(9);//只是用3x3的做了个例子
	Mat image = imread("D:/3.bmp", 1);//原图
	Mat image2;
	if (!image.data)
	{
		return -1;
	}
	//image3为image裁剪后，row=1100,col=950
	Mat image3(image, Rect(10, 10, 1100, 950));
	//image2为image3转灰度得到的图像，不知道里面存的值的大概范围
	cvtColor(image3, image2, CV_BGR2GRAY);
	int width = image2.cols;
	int height = image2.rows;
	//二值化
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			int index = i * width + j;
			if (image2.data[index] > 30)
			{
				image2.data[index] = 255;
			}
			else
			{
				image2.data[index] = 0;
			}
		}
	}
	imshow("灰度化后", image2);
	
	Mat grad_x;
	Mat grad_y;
	Mat dst;
	
	Sobel(image2, grad_x, CV_16S, 1, 0, 3);
	Sobel(image2, grad_y, CV_16S, 0, 1, 3);
	//Mat result;//去噪后
	//medianBlur(image2, result, 3);
	//Sobel(result, grad_x, CV_16S, 1, 0, 3);
	//Sobel(result, grad_y, CV_16S, 0, 1, 3);
	convertScaleAbs(grad_x, grad_x);
	convertScaleAbs(grad_y, grad_y);
	addWeighted(grad_x, 0.5, grad_y, 0.5, 0, dst);
	//imshow("边缘提取x后", grad_x);
	//imshow("边缘提取y后", grad_y);
	imshow("边缘提取后", dst);
	
	int flag=1;
	for (int x = 0; x < height; ++x)//遍历
	{
		for (int y = 0; y < width; ++y)
		{
			int index = x * width + y;
			if (int(dst.data[index]) == 0)//为黑色则跳过，且设置flag为0，代表一个图片已经分割完毕
			{
				flag = 0;
				continue;
			}
			else
			{
				if (flag == 0)//flag为0代表一个图片已经分割完毕,需要创建新矩阵存图
				{
					//创建新矩阵sub_image
				}
				while (IsRight(dst, x, y, width, height))//这个点在闭合轮廓上
				{
					//将sub_image里对应的点标记为白色
					//得到IsRight函数里i、j的值以沿着轮廓往下走
					//随机选择一个i、j作为下一个点
					//判断下个点是否在矩阵里了，如果是，则换一个点,如果全是，则退出while循环
					//如果不是，更新x，y
				}
				flag = 1;
			}



		}
	};

		//输出
		/*
		for (int i = 0; i < height; ++i)
		{
			for (int j = 0; j < width; ++j)
			{
				int index = i * width + j;
				printf("%d  ", int(dst.data[index]));
			}
		}*/
		


		//中值滤波 异常
		/*
		Mat img;//用于输出自己写的中值滤波
		image2.copyTo(img);
		///Mat result;//用于输出opencv自带的中值滤波
		for (int i = 140; i < image2.rows - 130; i++)
		{
			uchar* preptr = image2.ptr(i - 1);//(i,j)是要改变的像素坐标点
			uchar* ptr = image2.ptr(i);
			uchar* nextptr = image2.ptr(i + 1);
			uchar* imgptr = img.ptr(i);
			for (int j = 140; j < image2.cols - 130; j++)
			{
				grayv[0] = (preptr[j - 1]);
				grayv[1] = (preptr[j]);
				grayv[2] = (preptr[j + 1]);
				grayv[3] = (ptr[j - 1]);
				grayv[4] = (ptr[j]);
				grayv[5] = (ptr[j + 1]);
				grayv[6] = (nextptr[j - 1]);
				grayv[7] = (nextptr[j]);
				grayv[8] = (nextptr[j + 1]);
				sort(grayv.begin(), grayv.end());
				imgptr[j] = int(grayv[4]);
			}

		}
		//medianBlur(image2, result, 3);
		imshow("原图", image2);
		imshow("中值滤波后", img);
		//imshow("opencv自带滤波", result);
		*/
		waitKey(0);
	return 0;
}