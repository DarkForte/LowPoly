#pragma once

#include "point.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

struct Triangle
{
    Point points[3];
    Triangle(){}

    CUDA_HOSTDEV
    Triangle(Point a, Point b, Point c);

    CUDA_HOSTDEV
    Point center();

};
