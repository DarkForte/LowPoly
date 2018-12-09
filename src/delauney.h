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
void DelauneyGPU(int rows, int cols);
void getGradGPU(cv::Mat &img);
void selectVerticesGPU(float edgeThresh, float edgeP, float nonEdgeP, float boundP, int rows, int cols);
cv::Mat drawTriangleGPU(cv::Mat& img);
void fakeInit();