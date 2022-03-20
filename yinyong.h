#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

struct workpiece//工件结构
{
	Point centroid;//质心
	int zhouchang;//周长（边缘点个数）
	Point up_left;//外接矩形左上点
	Point dwn_rit;//外接矩形右下点
	float m_m_ratio;//到质心最大最小距离比
	Point near;//最近点
	Point far;//最远点
	vector<Point> edge;//边缘点集
	/*下面的不知道有没有用*/
	float k_s;//短距斜率
	float k_l;//长距斜率
} ;