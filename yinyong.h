#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
struct workpiece//工件结构
{
	//Point center;//重心
	int x_min=0;
	int x_max=0;
	int y_min=0;
	int y_max=0;
	int area=0;//面积
} Gj;