
#include<opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <numeric>
#include "yinyong.h"

using namespace cv;
using namespace std;
/**
 * @brief 区域生长算法，输入图像应为灰度图像
 * @param srcImage 区域生长的源图像
 * @param pt 区域生长点
 * @param ch1Thres 通道的生长限制阈值，临近像素符合±chxThres范围内才能进行生长
 * @param ch1LowerBind 通道的最小值阈值
 * @param ch1UpperBind 通道的最大值阈值，在这个范围外即使临近像素符合±chxThres也不能生长
 * @return 生成的区域图像（二值类型）
 */


Mat RegionGrow(Mat srcImage, Point pt, int ch1Thres, int ch1LowerBind , int ch1UpperBind, vector<Point>* pDset)
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
						(*pDset).push_back(pToGrowing);//保存被生长的点
					}
				}
			}
		}
	}
	return growImage.clone();
}

/*
Imchange
测试保存好的点是否正确
根据保存好的vector转换成图
*/
void Imchange(vector<Point> Dset,Mat* p_image,int width,int height)
{
	int x = 0;
	int y = 0;
	Mat testIm = Mat::zeros(width, height, CV_8UC1);
	for (int i = 0; i < Dset.size(); i++)
	{
		x = Dset[i].x;
		y = Dset[i].y;
		testIm.at<uchar>(y, x) = 255;
		(*p_image).at<uchar>(y, x) = 0;
	}
	//imwrite("D:/test_vector.jpg", testIm);
	//imwrite("D:/after_change.jpg", *(p_image));
	//imshow("ceshi", testIm);
	cv::waitKey(0);
}

bool cmpy_x(cv::Point const& a, cv::Point const& b)
{
	return a.x < b.x;
}
/*
* 找外接矩形的几何参数
*/
void rect_Find(struct workpiece* gj,int cut_num)
{
	int i = 0;//记录当前找的是第几个工件
	for (i = 0;i<cut_num; i++)
	{
		point_far_near(gj+i);//找该工件到质心最远和最近的点

	}
}

/*
* 返回离质心最近的点
*/

void point_far_near(struct workpiece* Gj)
{
	int  maxi,mini;//最远最近点的索引
	vector<float> dist;
	int dx, dy;
	for (int i = 0; i < (*Gj).edge.size(); i++)
	{
		dx = (*Gj).edge[i].x - (*Gj).centroid.x;
		dy = (*Gj).edge[i].y - (*Gj).centroid.y;
		dist[i] = sqrt(pow(dx, 2) + pow(dy, 2));
	}
	std::vector<float>::iterator max = max_element(dist.begin(), dist.end());
	maxi = std::distance(std::begin(dist), max);//最大值的下标
	std::vector<float>::iterator min = min_element(dist.begin(), dist.end());
	mini = std::distance(std::begin(dist), min);//最小值的下标
	(*Gj).far = (*Gj).edge[maxi];//最远点
	(*Gj).near = (*Gj).edge[mini];//最近点
}


/*
* 返回长短轴两个斜率
*/

/*
*主函数
*/

int main()
{
	Mat image = imread("D:/集合无粘连.bmp", 1);//原图
	Mat image1;
	//vector<Point> DotSet;
	vector<vector<Point>> after_cut;
	after_cut.resize(20);
	struct workpiece gj[20];//最多20个工件信息的结构数组

	if (!image.data)
	{
		return -1;
	}
	//image3为image裁剪后，row=130,col=110
	Mat image3(image, Rect(1, 1, 570, 570));
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
	Mat grad_x;
	Mat grad_y;
	Mat dst;
	Mat result = image2.clone();
	Sobel(result, grad_x, CV_16S, 1, 0, 3);
	Sobel(result, grad_y, CV_16S, 0, 1, 3);
	convertScaleAbs(grad_x, grad_x);
	convertScaleAbs(grad_y, grad_y);
	addWeighted(grad_x, 0.5, grad_y, 0.5, 0, image2);
	imwrite("D:/kangkang.jpg", image2);
	//得到image2为二值化后的图像
	/*
	//看输出
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
	int cut_num=-1;//记录工件个数
	Point start;
	for (int x = 0; x < height; ++x)//遍历
	{
		for (int y = 0; y < width; ++y)
		{
			int index = x * width + y;//(x,y)点的index
			if (int(image2.data[index]) < 2)//(x,y)为黑色则跳过，且设置flag为0，代表开始搜寻下一个图片了
			{
				continue;
			}
			else//（x,y)为前景
			{
				cut_num++;
				int m = y;
				int n = x;
				start.x = m;
				start.y = n;
				//以当前点区域生长后，point被存到after_cut[cut_num]里
				
				//Mat result = RegionGrow(image2, start, 127, 2, 255, &(after_cut[cut_num]));
				Mat result = RegionGrow(image2, start, 150, 2, 255, &(after_cut[cut_num]));
				imwrite("D:/try.jpg", result);
				//Imchange(DotSet,&image2,width,height);
				Imchange(after_cut[cut_num], &image2, width, height);
			}
		}
	}
	//存储分割后工件的Mat
	vector<Mat> after_cut_Mat;
	after_cut_Mat.resize(20);

	for (int p = 0; p < cut_num+1; p++)
	{
		after_cut_Mat[p]= Mat::zeros(width, height, CV_8UC1);
		for (int i = 0; i < after_cut[p].size(); i++)
		{
			int a = after_cut[p][i].x;
			int b = after_cut[p][i].y;
			after_cut_Mat[p].at<uchar>(b, a) = 255;
		}
		sort(after_cut[p].begin(), after_cut[p].end(),cmpy_x);
		//gj[p].x_min = after_cut[p][0].x;
		//gj[p].x_max = after_cut[p][].x;
		
		imwrite("D:/gongjian.jpg", after_cut_Mat[p]);//检验是否成功
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
}//保护保护