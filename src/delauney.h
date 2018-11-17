#pragma once

#include <cuda.h>
#include <vector>
#include "point.h"
#include "triangle.h"
using namespace std;

vector<Triangle> DelauneyCPU(vector<Point> &seeds, vector<int> &owner, int rows, int cols);
vector<Triangle> DelauneyGPU(vector<Point> &seeds, vector<int> &owner, int rows, int cols);