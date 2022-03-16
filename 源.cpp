
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
	
	vector<int>grayv(9);//ֻ����3x3�����˸�����
	Mat image = imread("D:/3.bmp", 1);//ԭͼ
	Mat image2;
	if (!image.data)
	{
		return -1;
	}
	//image3Ϊimage�ü���row=1100,col=950
	Mat image3(image, Rect(10, 10, 1100, 950));
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
	imshow("�ҶȻ���", image2);
	
	Mat grad_x;
	Mat grad_y;
	Mat dst;
	
	Sobel(image2, grad_x, CV_16S, 1, 0, 3);
	Sobel(image2, grad_y, CV_16S, 0, 1, 3);
	//Mat result;//ȥ���
	//medianBlur(image2, result, 3);
	//Sobel(result, grad_x, CV_16S, 1, 0, 3);
	//Sobel(result, grad_y, CV_16S, 0, 1, 3);
	convertScaleAbs(grad_x, grad_x);
	convertScaleAbs(grad_y, grad_y);
	addWeighted(grad_x, 0.5, grad_y, 0.5, 0, dst);
	//imshow("��Ե��ȡx��", grad_x);
	//imshow("��Ե��ȡy��", grad_y);
	imshow("��Ե��ȡ��", dst);
	
	int flag=1;
	for (int x = 0; x < height; ++x)//����
	{
		for (int y = 0; y < width; ++y)
		{
			int index = x * width + y;
			if (int(dst.data[index]) == 0)//Ϊ��ɫ��������������flagΪ0������һ��ͼƬ�Ѿ��ָ����
			{
				flag = 0;
				continue;
			}
			else
			{
				if (flag == 0)//flagΪ0����һ��ͼƬ�Ѿ��ָ����,��Ҫ�����¾����ͼ
				{
					//�����¾���sub_image
				}
				while (IsRight(dst, x, y, width, height))//������ڱպ�������
				{
					//��sub_image���Ӧ�ĵ���Ϊ��ɫ
					//�õ�IsRight������i��j��ֵ����������������
					//���ѡ��һ��i��j��Ϊ��һ����
					//�ж��¸����Ƿ��ھ������ˣ�����ǣ���һ����,���ȫ�ǣ����˳�whileѭ��
					//������ǣ�����x��y
				}
				flag = 1;
			}



		}
	};

		//���
		/*
		for (int i = 0; i < height; ++i)
		{
			for (int j = 0; j < width; ++j)
			{
				int index = i * width + j;
				printf("%d  ", int(dst.data[index]));
			}
		}*/
		


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
		waitKey(0);
	return 0;
}