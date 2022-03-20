
#include<opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <numeric>
#include "yinyong.h"

using namespace cv;
using namespace std;
/**
 * @brief ���������㷨������ͼ��ӦΪ�Ҷ�ͼ��
 * @param srcImage ����������Դͼ��
 * @param pt ����������
 * @param ch1Thres ͨ��������������ֵ���ٽ����ط��ϡ�chxThres��Χ�ڲ��ܽ�������
 * @param ch1LowerBind ͨ������Сֵ��ֵ
 * @param ch1UpperBind ͨ�������ֵ��ֵ���������Χ�⼴ʹ�ٽ����ط��ϡ�chxThresҲ��������
 * @return ���ɵ�����ͼ�񣨶�ֵ���ͣ�
 */


Mat RegionGrow(Mat srcImage, Point pt, int ch1Thres, int ch1LowerBind , int ch1UpperBind, vector<Point>* pDset)
{
	Point pToGrowing;                       //��������λ��
	int pGrowValue = 0;                             //��������Ҷ�ֵ
	Scalar pSrcValue = 0;                               //�������Ҷ�ֵ
	Scalar pCurValue = 0;                               //��ǰ������Ҷ�ֵ
	Mat growImage = Mat::zeros(srcImage.size(), CV_8UC1);   //����һ���հ��������Ϊ��ɫ
	//��������˳������
	int DIR[8][2] = { {-1,-1}, {0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0} };
	vector<Point> growPtVector;                     //������ջ
	growPtVector.push_back(pt);                         //��������ѹ��ջ��
	growImage.at<uchar>(pt.y, pt.x) = 255;              //���������
	pSrcValue = srcImage.at<uchar>(pt.y, pt.x);         //��¼������ĻҶ�ֵ

	while (!growPtVector.empty())                       //����ջ��Ϊ��������
	{
		pt = growPtVector.back();                       //ȡ��һ��������
		growPtVector.pop_back();

		//�ֱ�԰˸������ϵĵ��������
		for (int i = 0; i < 9; ++i)
		{
			pToGrowing.x = pt.x + DIR[i][0];
			pToGrowing.y = pt.y + DIR[i][1];
			//����Ƿ��Ǳ�Ե��
			if (pToGrowing.x < 0 || pToGrowing.y < 0 ||
				pToGrowing.x >(srcImage.cols - 1) || (pToGrowing.y > srcImage.rows - 1))
				continue;

			pGrowValue = growImage.at<uchar>(pToGrowing.y, pToGrowing.x);       //��ǰ��������ĻҶ�ֵ
			pSrcValue = srcImage.at<uchar>(pt.y, pt.x);
			if (pGrowValue == 0)                    //�����ǵ㻹û�б�����
			{
				pCurValue = srcImage.at<uchar>(pToGrowing.y, pToGrowing.x);
				if (pCurValue[0] <= ch1UpperBind && pCurValue[0] >= ch1LowerBind)
				{
					if (abs(pSrcValue[0] - pCurValue[0]) < ch1Thres)                   //����ֵ��Χ��������
					{
						growImage.at<uchar>(pToGrowing.y, pToGrowing.x) = 255;      //���Ϊ��ɫ
						growPtVector.push_back(pToGrowing);                 //����һ��������ѹ��ջ��
						(*pDset).push_back(pToGrowing);//���汻�����ĵ�
					}
				}
			}
		}
	}
	return growImage.clone();
}

/*
Imchange
���Ա���õĵ��Ƿ���ȷ
���ݱ���õ�vectorת����ͼ
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
* ����Ӿ��εļ��β���
*/
void rect_Find(struct workpiece* gj,int cut_num)
{
	int i = 0;//��¼��ǰ�ҵ��ǵڼ�������
	for (i = 0;i<cut_num; i++)
	{
		point_far_near(gj+i);//�Ҹù�����������Զ������ĵ�

	}
}

/*
* ��������������ĵ�
*/

void point_far_near(struct workpiece* Gj)
{
	int  maxi,mini;//��Զ����������
	vector<float> dist;
	int dx, dy;
	for (int i = 0; i < (*Gj).edge.size(); i++)
	{
		dx = (*Gj).edge[i].x - (*Gj).centroid.x;
		dy = (*Gj).edge[i].y - (*Gj).centroid.y;
		dist[i] = sqrt(pow(dx, 2) + pow(dy, 2));
	}
	std::vector<float>::iterator max = max_element(dist.begin(), dist.end());
	maxi = std::distance(std::begin(dist), max);//���ֵ���±�
	std::vector<float>::iterator min = min_element(dist.begin(), dist.end());
	mini = std::distance(std::begin(dist), min);//��Сֵ���±�
	(*Gj).far = (*Gj).edge[maxi];//��Զ��
	(*Gj).near = (*Gj).edge[mini];//�����
}


/*
* ���س���������б��
*/

/*
*������
*/

int main()
{
	Mat image = imread("D:/������ճ��.bmp", 1);//ԭͼ
	Mat image1;
	//vector<Point> DotSet;
	vector<vector<Point>> after_cut;
	after_cut.resize(20);
	struct workpiece gj[20];//���20��������Ϣ�Ľṹ����

	if (!image.data)
	{
		return -1;
	}
	//image3Ϊimage�ü���row=130,col=110
	Mat image3(image, Rect(1, 1, 570, 570));
	//image2Ϊimage3ת�Ҷȵõ���ͼ�񣬲�֪��������ֵ�Ĵ�ŷ�Χ
	cvtColor(image3, image1, CV_BGR2GRAY);
	//cvtColor(image, image1, CV_BGR2GRAY);
	Mat image2;//ȥ���
	medianBlur(image1, image2, 3);
	int width = image2.cols;
	int height = image2.rows;
	//��ֵ��
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
	imwrite("D:/ǰ��Ϊɶɫ.jpg", image2);
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
	//�õ�image2Ϊ��ֵ�����ͼ��
	/*
	//�����
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
	int cut_num=-1;//��¼��������
	Point start;
	for (int x = 0; x < height; ++x)//����
	{
		for (int y = 0; y < width; ++y)
		{
			int index = x * width + y;//(x,y)���index
			if (int(image2.data[index]) < 2)//(x,y)Ϊ��ɫ��������������flagΪ0������ʼ��Ѱ��һ��ͼƬ��
			{
				continue;
			}
			else//��x,y)Ϊǰ��
			{
				cut_num++;
				int m = y;
				int n = x;
				start.x = m;
				start.y = n;
				//�Ե�ǰ������������point���浽after_cut[cut_num]��
				
				//Mat result = RegionGrow(image2, start, 127, 2, 255, &(after_cut[cut_num]));
				Mat result = RegionGrow(image2, start, 150, 2, 255, &(after_cut[cut_num]));
				imwrite("D:/try.jpg", result);
				//Imchange(DotSet,&image2,width,height);
				Imchange(after_cut[cut_num], &image2, width, height);
			}
		}
	}
	//�洢�ָ�󹤼���Mat
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
		
		imwrite("D:/gongjian.jpg", after_cut_Mat[p]);//�����Ƿ�ɹ�
	}
	


	/*
	//�Լ�����ʼ�㣬�ɹ�
	Point start;
	start.x = 55;
	start.y = 65;
	Mat result = RegionGrow(image2, start, 127, 2, 255);
	imshow("try", result);
	*/

	/*
	vector<int>grayv(9);//ֻ����3x3�����˸�����
	Mat image = imread("D:/Image_0022.bmp", 1);//ԭͼ
	Mat image2;
	if (!image.data)
	{
		return -1;
	}
	//image3Ϊimage�ü���row=1100,col=950
	Mat image3(image, Rect(600,570, 130, 110));
	//image2Ϊimage3ת�Ҷȵõ���ͼ�񣬲�֪��������ֵ�Ĵ�ŷ�Χ
	cvtColor(image3, image2, CV_BGR2GRAY);
	int width = image2.cols;
	int height = image2.rows;
	//��ֵ��
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
	imshow("��ֵ����", image2);
	*/

	/*
	Mat grad_x;
	Mat grad_y;
	Mat dst_before;
	Mat dst;


	//Sobel(image2, grad_x, CV_16S, 1, 0, 3);
	//Sobel(image2, grad_y, CV_16S, 0, 1, 3);
	Mat result;//ȥ���
	medianBlur(image2, result, 3);
	Sobel(result, grad_x, CV_16S, 1, 0, 3);
	Sobel(result, grad_y, CV_16S, 0, 1, 3);
	convertScaleAbs(grad_x, grad_x);
	convertScaleAbs(grad_y, grad_y);
	addWeighted(grad_x, 0.5, grad_y, 0.5, 0, dst_before);
	//imshow("��Ե��ȡx��", grad_x);
	//imshow("��Ե��ȡy��", grad_y);
	imshow("��Ե��ȡ��", dst_before);
	//���
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
	dst = dst_before.clone();//dst���ָ���
	int** location;
	int flag=1;

	//Mat each_sub_image[100];//���õ��Ľ��
	Mat each_sub_image= cv::Mat::zeros(height, width, CV_32F);
	Mat sub_image;//��ʱ���
	int image_num = -1;
	int turn2black[200] = { 0 };
	int black_num=0;
	for (int x = 0; x < height; ++x)//����
	{
		for (int y = 0; y < width; ++y)
		{
			int index = x * width + y;//(x,y)���index
			//(x,y)Ϊ��ɫ��������������flagΪ0������ʼ��Ѱ��һ��ͼƬ��
			if (int(dst.data[index]) == 0)//(x,y)Ϊ��ɫ��������������flagΪ0������ʼ��Ѱ��һ��ͼƬ��
			{
				flag = 0;
				continue;
			}
			else//��x,y)Ϊǰ��
			{
				if (flag == 0)//flagΪ0��������Ѱ����һ��ͼƬ��
				{
					//black_num = 0;
					//���ҵ��ڶ���ͼ�ο�ʼ������һ��ͼƬ��������image_num��ͼ���mat��
					if (image_num != -1)
					{
						//each_sub_image[image_num] = sub_image.clone();
						each_sub_image = sub_image.clone();
						imwrite("D:/SubImage2.jpg", each_sub_image);
					}
					image_num++;
				}
				//������ʱmat����ͼ������ǵ��Ƿ���ʹ�
				sub_image = cv::Mat::zeros(height, width, CV_32F);//ȫ��
				Mat flag_image = cv::Mat::zeros(height, width, CV_32F);//ȫΪ0
				//��m,n)Ϊ��ǰ���ʵĵ������
				int m = x;
				int n = y;
				//location���أ�m,n)�����ڵĵ����Բ��
				location = IsRight(dst, m, n, width, height);
				int index_sub = m * width + n;
				while (location[0][0]!=0)//������ڱպ�������
				{
					//��sub_image��(m,n)��Ӧ�ĵ���Ϊ��ɫ,��flag_image��(m,n)��Ӧ�ĵ���Ϊ1
					sub_image.data[index_sub] = 255;
					flag_image.data[index_sub] = 1;
					//index�浽turn2black�һ��ͼ�α�������֮��Ѷ�Ӧ��ȫ����ɺ�ɫ
					turn2black[black_num] = index_sub;
					black_num++;
					int num_max = location[0][0];//���м�����ѡ��
					int i = 0;
					int j = 0;
					int num = 0;
					//�ж��¸�����flag_image��Ķ�Ӧֵ�Ƿ�Ϊ1��Ϊ1�Ļ���һ��i��j
					while (flag_image.data[(m + i) * width + n + j] == 1 && num<=num_max)
					{
						//ѡ��һ����һ��i��j
						num++;
						if (num <= num_max)
						{
							i = location[0][num];
							j = location[1][num];
						}
					}
					if (num > num_max)
						break;
					//����m��nΪ���ڽӵ�
					m = m + i;
					n = n + j;
					location = IsRight(dst, m, n, width, height);
					index_sub = m * width + n;
				}
				//�������������ĵ���dst����Ϊ����
				for (int q = 0; q < black_num + 1; q++)
				{
					int black_index;
					black_index = turn2black[q];
					dst.data[black_index] = 0;
					//��turn2black����
					turn2black[q] = 0;
				}
				black_num = 0;
				//imwrite("D:/SubImage2.jpg", each_sub_image);
				flag = 1;
			}
		}
	}
	*/
	//��ֵ�˲� �쳣
	/*
	Mat img;//��������Լ�д����ֵ�˲�
	image2.copyTo(img);
	///Mat result;//�������opencv�Դ�����ֵ�˲�
	for (int i = 140; i < image2.rows - 130; i++)
	{
		uchar* preptr = image2.ptr(i - 1);//(i,j)��Ҫ�ı�����������
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
	imshow("ԭͼ", image2);
	imshow("��ֵ�˲���", img);
	//imshow("opencv�Դ��˲�", result);
	*/

	cv::waitKey(0);
	return 0;
}//��������