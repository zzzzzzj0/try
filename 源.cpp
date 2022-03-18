
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
	//location_thisΪ��ά����ͷ
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
						location_this[0][k] = i;//0ά��¼i
						location_this[1][k] = j;//1ά��¼j
					}
				}
			}
		}
	}
	if (k > 2)//������ͷ����Ƿ��ڱպ�������
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
 * @brief ���������㷨������ͼ��ӦΪ�Ҷ�ͼ��
 * @param srcImage ����������Դͼ��
 * @param pt ����������
 * @param ch1Thres ͨ��������������ֵ���ٽ����ط��ϡ�chxThres��Χ�ڲ��ܽ�������
 * @param ch1LowerBind ͨ������Сֵ��ֵ
 * @param ch1UpperBind ͨ�������ֵ��ֵ���������Χ�⼴ʹ�ٽ����ط��ϡ�chxThresҲ��������
 * @return ���ɵ�����ͼ�񣨶�ֵ���ͣ�
 */


Mat RegionGrow(Mat srcImage, Point pt, int ch1Thres, int ch1LowerBind , int ch1UpperBind )
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
					}
				}
			}
		}
	}
	return growImage.clone();
}

int main()
{
	Mat image = imread("D:/3.bmp", 1);//ԭͼ
	Mat image1;
	if (!image.data)
	{
		return -1;
	}
	//image3Ϊimage�ü���row=130,col=110
	Mat image3(image, Rect(100, 150, 800, 800));
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
	//�õ�image2Ϊ��ֵ�����ͼ��
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
	for (int x = 0; x < height; ++x)//����
	{
		for (int y = 0; y < width; ++y)
		{
			int index = x * width + y;//(x,y)���index
			if (int(image2.data[index]) < 2 )//(x,y)Ϊ��ɫ��������������flagΪ0������ʼ��Ѱ��һ��ͼƬ��
			{
				continue;
			}
			else//��x,y)Ϊǰ��
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
}