#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

struct workpiece//�����ṹ
{
	Point centroid;//����
	int zhouchang;//�ܳ�����Ե�������
	Point up_left;//��Ӿ������ϵ�
	Point dwn_rit;//��Ӿ������µ�
	float m_m_ratio;//�����������С�����
	Point near;//�����
	Point far;//��Զ��
	vector<Point> edge;//��Ե�㼯
	/*����Ĳ�֪����û����*/
	float k_s;//�̾�б��
	float k_l;//����б��
} ;