
#include<opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include<time.h>
using namespace cv;
using namespace std;


/*
int** IsRight(Mat image, int x, int y, int width, int height)
{
	int i, j, k = 0;
	int index = 0;
	int** location_this;
	//location_this为二维数组头
	location_this = (int**)malloc(2 * sizeof(int*));
	if (location_this == NULL)
	{
		printf("no!");
		exit(1);
	}
	for (int i = 0; i < 2; i++) 
	{
		location_this[i] = (int*)malloc(10 * sizeof(int));
		if (location_this[i] == NULL)
		{
			printf("no!");
			exit(1);
		}
	}
	for (i = -1; i <= 1; i++)
	{
		if (x + i < width && x + i >= 0)
		{
			for (j = -1; j <= 1; j++)
			{
				if (y + j < height && y + j >= 0)
				{
					index = (x+i) * width + y+j;
					if (image.data[index] != 0)
					{
						k++;
						location_this[0][k] = i;//0维记录i
						location_this[1][k] = j;//1维记录j
					}
				}
			}
		}
	}
	if (k > 2)//以数组头标记是否在闭合轮廓上
	{
		location_this[0][0] = k;
		location_this[1][0] = k;
	}
	else
	{
		location_this[0][0] = 0;
		location_this[1][0] = 0;
	}
	return location_this;
}
*/
/**
 * @brief 区域生长算法，输入图像应为灰度图像
 * @param srcImage 区域生长的源图像
 * @param pt 区域生长点
 * @param ch1Thres 通道的生长限制阈值，临近像素符合±chxThres范围内才能进行生长
 * @param ch1LowerBind 通道的最小值阈值
 * @param ch1UpperBind 通道的最大值阈值，在这个范围外即使临近像素符合±chxThres也不能生长
 * @return 生成的区域图像（二值类型）
 */


Mat RegionGrow(Mat srcImage, Point pt, int ch1Thres, int ch1LowerBind , int ch1UpperBind )
{
	Point pToGrowing;                       //待生长点位置
	int pGrowValue = 0;                             //待生长点灰度值
	Scalar pSrcValue = 0;                               //生长起点灰度值
	Scalar pCurValue = 0;                               //当前生长点灰度值
	Mat growImage = Mat::zeros(srcImage.size(), CV_8UC1);   //创建一个空白区域，填充为黑色
	//生长方向顺序数据
	int DIR[8][2] = { {-1,-1}, {0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0} };
	vector<Point> growPtVector;                     //生长点栈
	growPtVector.push_back(pt);                         //将生长点压入栈中
	growImage.at<uchar>(pt.y, pt.x) = 255;              //标记生长点
	pSrcValue = srcImage.at<uchar>(pt.y, pt.x);         //记录生长点的灰度值

	while (!growPtVector.empty())                       //生长栈不为空则生长
	{
		pt = growPtVector.back();                       //取出一个生长点
		growPtVector.pop_back();

		//分别对八个方向上的点进行生长
		for (int i = 0; i < 9; ++i)
		{
			pToGrowing.x = pt.x + DIR[i][0];
			pToGrowing.y = pt.y + DIR[i][1];
			//检查是否是边缘点
			if (pToGrowing.x < 0 || pToGrowing.y < 0 ||
				pToGrowing.x >(srcImage.cols - 1) || (pToGrowing.y > srcImage.rows - 1))
				continue;

			pGrowValue = growImage.at<uchar>(pToGrowing.y, pToGrowing.x);       //当前待生长点的灰度值
			pSrcValue = srcImage.at<uchar>(pt.y, pt.x);
			if (pGrowValue == 0)                    //如果标记点还没有被生长
			{
				pCurValue = srcImage.at<uchar>(pToGrowing.y, pToGrowing.x);
				if (pCurValue[0] <= ch1UpperBind && pCurValue[0] >= ch1LowerBind)
				{
					if (abs(pSrcValue[0] - pCurValue[0]) < ch1Thres)                   //在阈值范围内则生长
					{
						growImage.at<uchar>(pToGrowing.y, pToGrowing.x) = 255;      //标记为白色
						growPtVector.push_back(pToGrowing);                 //将下一个生长点压入栈中
					}
				}
			}
		}
	}
	return growImage.clone();
}

int main()
{
	Mat image = imread("D:/3.bmp", 1);//原图
	Mat image1;
	if (!image.data)
	{
		return -1;
	}
	//image3为image裁剪后，row=130,col=110
	Mat image3(image, Rect(100, 150, 800, 800));
	//image2为image3转灰度得到的图像，不知道里面存的值的大概范围
	cvtColor(image3, image1, CV_BGR2GRAY);
	//cvtColor(image, image1, CV_BGR2GRAY);
	Mat image2;//去噪后
	medianBlur(image1, image2, 3);
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
				image2.data[index] = 0;
			}
			else
			{
				image2.data[index] = 255;
			}
		}
	}
	imwrite("D:/前景为啥色.jpg", image2);
	//得到image2为二值化后的图像
	/*
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			int index = i * width + j;
			printf("%d  ", int(image2.data[index]));
		}
		printf("\n\n");
	}
	*/
	
	Point start;
	for (int x = 0; x < height; ++x)//遍历
	{
		for (int y = 0; y < width; ++y)
		{
			int index = x * width + y;//(x,y)点的index
			if (int(image2.data[index]) < 2 )//(x,y)为黑色则跳过，且设置flag为0，代表开始搜寻下一个图片了
			{
				continue;
			}
			else//（x,y)为前景
			{
				int m = y;
				int n = x;
				start.x = m ;
				start.y = n ;
				Mat result = RegionGrow(image2, start, 127, 2, 255);
				imwrite("D:/try.jpg", result);
			}
		}
	}
	
	/*
	//自己设起始点，成功
	Point start;
	start.x = 55;
	start.y = 65;
	Mat result = RegionGrow(image2, start, 127, 2, 255);
	imshow("try", result);
	*/

	/*
	vector<int>grayv(9);//只是用3x3的做了个例子
	Mat image = imread("D:/Image_0022.bmp", 1);//原图
	Mat image2;
	if (!image.data)
	{
		return -1;
	}
	//image3为image裁剪后，row=1100,col=950
	Mat image3(image, Rect(600,570, 130, 110));
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
	imshow("二值化后", image2);
	*/

		
		


		/*
		Mat grad_x;
		Mat grad_y;
		Mat dst_before;
		Mat dst;


		//Sobel(image2, grad_x, CV_16S, 1, 0, 3);
		//Sobel(image2, grad_y, CV_16S, 0, 1, 3);
		Mat result;//去噪后
		medianBlur(image2, result, 3);
		Sobel(result, grad_x, CV_16S, 1, 0, 3);
		Sobel(result, grad_y, CV_16S, 0, 1, 3);
		convertScaleAbs(grad_x, grad_x);
		convertScaleAbs(grad_y, grad_y);
		addWeighted(grad_x, 0.5, grad_y, 0.5, 0, dst_before);
		//imshow("边缘提取x后", grad_x);
		//imshow("边缘提取y后", grad_y);
		imshow("边缘提取后", dst_before);
		//输出
		/*
		for (int i = 9; i < 23; ++i)
		{
			for (int j = 0; j < width; ++j)
			{
					int index = i * width + j;
					printf("%d  ", int(dst_before.data[index]));
			}
			printf("\n\n");
		}
		*/

		/*
		dst = dst_before.clone();//dst给分割用
		int** location;
		int flag=1;

		//Mat each_sub_image[100];//最后得到的结果
		Mat each_sub_image= cv::Mat::zeros(height, width, CV_32F);
		Mat sub_image;//临时存放
		int image_num = -1;
		int turn2black[200] = { 0 };
		int black_num=0;
		for (int x = 0; x < height; ++x)//遍历
		{
			for (int y = 0; y < width; ++y)
			{
				int index = x * width + y;//(x,y)点的index
				//(x,y)为黑色则跳过，且设置flag为0，代表开始搜寻下一个图片了
				if (int(dst.data[index]) == 0)//(x,y)为黑色则跳过，且设置flag为0，代表开始搜寻下一个图片了
				{
					flag = 0;
					continue;
				}
				else//（x,y)为前景
				{
					if (flag == 0)//flag为0代表是搜寻到下一张图片了
					{
						//black_num = 0;
						//从找到第二个图形开始，把上一张图片拷贝到第image_num个图像的mat里
						if (image_num != -1)
						{
							//each_sub_image[image_num] = sub_image.clone();
							each_sub_image = sub_image.clone();
							imwrite("D:/SubImage2.jpg", each_sub_image);
						}
						image_num++;
					}
					//创建临时mat放子图案并标记点是否访问过
					sub_image = cv::Mat::zeros(height, width, CV_32F);//全黑
					Mat flag_image = cv::Mat::zeros(height, width, CV_32F);//全为0
					//（m,n)为当前访问的点的坐标
					int m = x;
					int n = y;
					//location返回（m,n)领域内的点的相对差距
					location = IsRight(dst, m, n, width, height);
					int index_sub = m * width + n;
					while (location[0][0]!=0)//这个点在闭合轮廓上
					{
						//将sub_image里(m,n)对应的点标记为白色,将flag_image里(m,n)对应的点标记为1
						sub_image.data[index_sub] = 255;
						flag_image.data[index_sub] = 1;
						//index存到turn2black里，一个图形遍历结束之后把对应点全部变成黑色
						turn2black[black_num] = index_sub;
						black_num++;
						int num_max = location[0][0];//共有几个可选点
						int i = 0;
						int j = 0;
						int num = 0;
						//判断下个点在flag_image里的对应值是否为1，为1的话换一组i，j
						while (flag_image.data[(m + i) * width + n + j] == 1 && num<=num_max)
						{
							//选择一个下一个i、j
							num++;
							if (num <= num_max)
							{
								i = location[0][num];
								j = location[1][num];
							}
						}
						if (num > num_max)
							break;
						//更新m，n为其邻接点
						m = m + i;
						n = n + j;
						location = IsRight(dst, m, n, width, height);
						index_sub = m * width + n;
					}
					//把上述遍历过的点在dst里置为背景
					for (int q = 0; q < black_num + 1; q++)
					{
						int black_index;
						black_index = turn2black[q];
						dst.data[black_index] = 0;
						//把turn2black清零
						turn2black[q] = 0;
					}
					black_num = 0;
					//imwrite("D:/SubImage2.jpg", each_sub_image);
					flag = 1;
				}
			}
		}
		*/
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
	
		cv::waitKey(0);
	return 0;
}