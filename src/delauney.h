#pragma once

#include <cuda.h>
#include <vector>
#include "point.h"
#include "triangle.h"

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;

vector<Triangle> DelauneyCPU(vector<Point> &seeds, vector<int> &owner, int rows, int cols);
vector<Triangle> DelauneyGPU(Point* owner, int rows, int cols);
cv::Mat getGradGPU(cv::Mat &img);