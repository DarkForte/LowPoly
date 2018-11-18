#pragma once

// If it is being compiled by nvcc, __CUDACC__ with be defined and we can use __host__ __device__.
// Otherwise, it means that it is compiled by g++, g++ does not recognize __host__ so it will complain.

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

struct Point
{
    int x, y;
    CUDA_HOSTDEV
    Point(){}
    CUDA_HOSTDEV
    Point(int _x, int _y);
};

CUDA_HOSTDEV
int dist(const Point &a, const Point &b);

CUDA_HOSTDEV
Point operator + (const Point &a, const Point &b);

CUDA_HOSTDEV
Point operator * (const Point &a, int b);

CUDA_HOSTDEV
Point operator / (const Point &a, int b);