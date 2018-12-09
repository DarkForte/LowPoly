#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <ctime>
#include "triangle.h"
#include <vector>
#include <stdio.h>
#include <iostream>
#include "delauney.h"
#include "cycleTimer.h"
#include <curand.h>
#include <curand_kernel.h>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
using namespace std;

uint8_t* device_dummy = NULL;
uint8_t* device_img = NULL;
uint8_t* device_img_gray = NULL;
uint8_t* device_tri_img = NULL;
float* device_grad = NULL;
Point* device_seeds = NULL;
Point* device_ownerMap = NULL;
int* device_owner = NULL;
Triangle* device_triangles = NULL;
int num_triangles;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int NextPower2_CPU(int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

int Index_CPU(Point p, int col)
{
    return p.y * col + p.x;
}

__device__ __inline__ bool InBound(Point p, int row, int col)
{
    return (0 <= p.x && p.x < col && 0<=p.y && p.y < row);
}

__device__ __inline__ int Index(Point p, int col)
{
    return p.y * col + p.x;
}

__global__ void voronoi_kernel(Point* device_owner, int stepsize, int rows, int cols)
{   
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= cols || r >= rows)
        return;

    Point dir[] = {Point(0,1), Point(0, -1), Point(1, 0), Point(-1, 0),
                    Point(1,1), Point(1, -1), Point(-1, 1), Point(-1,-1)};

    Point now_point(c, r);
    for(Point now_dir: dir)
    {
        Point now_looking = now_point + now_dir * stepsize;
        if(!InBound(now_looking, rows, cols))
            continue;

        if(device_owner[Index(now_looking, cols)].isInvalid())
            continue;

        int cand_dist = dist(device_owner[Index(now_looking, cols)], now_point);
        Point now_owner = device_owner[Index(now_point, cols)];

        if(now_owner.isInvalid() || cand_dist < dist(now_owner, now_point))
            device_owner[Index(now_point, cols)] = device_owner[Index(now_looking, cols)];
    }
}

// Color = a seed
__device__ void count_colors(Point now_point, Point device_owner[], Point colors[4], int &numColors, int cols)
{
    Point neighbor_dir[] = {Point(1, 0), Point(0, 1), Point(1, 1)};

    colors[0] = device_owner[Index(now_point, cols)];
    numColors = 1;
    for(Point now_dir: neighbor_dir)
    {
        Point next_point = now_point + now_dir;
        Point newColor = device_owner[Index(next_point, cols)];
        bool exist = false;
        for (int i = 0; i < numColors; i++)
        {
            if (newColor == colors[i])
            {
                exist = true;
                break;
            }
        }
        if (!exist)
        {
            colors[numColors] = newColor;
            numColors ++;
        }
    }
}

// Parallelize by pixel, tell the caller the #triangle at (c, r)
__global__ void count_triangle_kernel(Point* device_owner, int rows, int cols, int* triangle_count)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= cols-1 || r >= rows-1)
        return;

    Point now_point(c, r);

    Point colors[4] = {Point(-1, -1), Point(-1, -1), Point(-1, -1), Point(-1, -1)};
    int numColors;
    count_colors(now_point, device_owner, colors, numColors, cols);

    if(numColors == 3)
        triangle_count[Index(now_point, cols)] = 1;
    else if(numColors == 4)
        triangle_count[Index(now_point, cols)] = 2;
    else
        triangle_count[Index(now_point, cols)] = 0;
}

// put triangle to device_triangles
__global__ void triangle_kernel(Point* device_owner, Triangle* device_triangles, int rows, int cols, int* device_sum_triangles)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= cols-1 || r >= rows-1)
        return;

    Point now_point(c, r);

    Point colors[4] = {Point(-1, -1), Point(-1, -1), Point(-1, -1), Point(-1, -1)};
    int numColors;
    count_colors(now_point, device_owner, colors, numColors, cols);

    int index = Index(now_point, cols);
    int prev_triangle_cnt = (index == 0)? 0: device_sum_triangles[index-1];

    if(numColors == 3)
    {
        Triangle triangle;
        for(int i=0; i<3; i++)
        {
            triangle.points[i] = colors[i];
        }

        device_triangles[prev_triangle_cnt] = triangle;
    }
    else if(numColors == 4)
    {
        Triangle triangle1(device_owner[Index(now_point, cols)],
                           device_owner[Index(now_point + Point(1, 0), cols)],
                           device_owner[Index(now_point + Point(0, 1), cols)]);

        Triangle triangle2(device_owner[Index(now_point + Point(1, 0), cols)],
                           device_owner[Index(now_point + Point(0, 1), cols)],
                           device_owner[Index(now_point + Point(1, 1), cols)]);

        device_triangles[prev_triangle_cnt] = triangle1;
        device_triangles[prev_triangle_cnt+1] = triangle2;
    }
    return;
}


__global__ void get_grad_kernel(uint8_t* device_img_gray, float* device_grad, int rows, int cols)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= cols || r >= rows)
        return;

    if (r > 0 && c > 0 && r < rows - 1 && c < cols - 1) // inside the image
    {
        float grad_x = -(float)device_img_gray[(r - 1) * cols + c - 1] + (float)device_img_gray[(r - 1) * cols + c + 1]
                       - 2 * (float)device_img_gray[r * cols + c - 1]  + 2 * (float)device_img_gray[r * cols + c + 1]
                       -(float)device_img_gray[(r + 1) * cols + c - 1] + (float)device_img_gray[(r + 1) * cols + c + 1];
        grad_x = abs(grad_x);

        float grad_y = -(float)device_img_gray[(r - 1) * cols + c - 1] - 2 * (float)device_img_gray[(r - 1) * cols + c] - (float)device_img_gray[(r - 1) * cols + c + 1]
                       +(float)device_img_gray[(r + 1) * cols + c - 1] + 2 * (float)device_img_gray[(r + 1) * cols + c] + (float)device_img_gray[(r + 1) * cols + c + 1];
        grad_y = abs(grad_y);

        float grad = grad_x / 2.0 + grad_y / 2.0;
        device_grad[r * cols + c] = grad;
    }

    else // set the boundary values to 0
    {
        device_grad[r * cols + c] = 0;
    }
}


__global__ void setup_rand_kernel(curandState *state)
{
    int idx = threadIdx.x * blockDim.x + threadIdx.y;
    curand_init(idx, 0, 0, &state[idx]);
}


__global__ void select_vertex_kernel(float* device_grad, Point* device_ownerMap, curandState *my_curandstate, float edgeThresh, float edgeP, float nonEdgeP, float boundP, int rows, int cols)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= cols || r >= rows)
        return;

    int idx = r * cols + c;
    int randIdx = threadIdx.x * blockDim.x + threadIdx.y;
    float randNum = curand_uniform(my_curandstate+randIdx);

    if (r > 0 && c > 0 && r < rows - 1 && c < cols - 1) // inside the image
    {
        if (device_grad[idx] > edgeThresh)
        {
            if (randNum <= edgeP)
            {
                Point p;
                p.x = c;
                p.y = r;
                device_ownerMap[idx] = p;
            }
        } else
        {
            if (randNum <= nonEdgeP)
            {
                Point p;
                p.x = c;
                p.y = r;
                device_ownerMap[idx] = p;
            }
        }
    }

    else // boundary
    {
        if (randNum <= boundP)
        {
            Point p;
            p.x = c;
            p.y = r;
            device_ownerMap[idx] = p;
        }
    }
}


__device__ __inline__ int signGPU(Point p1, Point p2, Point p3)
{
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}


__device__ bool PointInTriangleGPU(Point pt, Point v1, Point v2, Point v3)
{
    int d1, d2, d3;
    bool has_neg, has_pos;

    d1 = signGPU(pt, v1, v2);
    d2 = signGPU(pt, v2, v3);
    d3 = signGPU(pt, v3, v1);

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
}


__global__ void draw_triangle_kernel(Triangle* device_triangles, int num_triangles, uint8_t* device_img, uint8_t* device_tri_img, int rows, int cols)
{
    int triIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (triIdx >= num_triangles)
        return;

    Triangle tri = device_triangles[triIdx];
    int minX = min(tri.points[0].x, tri.points[1].x);
    minX = min(minX, tri.points[2].x);
    int minY = min(tri.points[0].y, tri.points[1].y);
    minY = min(minY, tri.points[2].y);    
    int maxX = max(tri.points[0].x, tri.points[1].x);
    maxX = max(maxX, tri.points[2].x);
    int maxY = max(tri.points[0].y, tri.points[1].y);
    maxY = max(maxY, tri.points[2].y);  
    Point p = tri.center();
    int imgIdx = (p.y * cols + p.x) * 3;

    for (int r = minY; r < maxY; ++r)
    {
        for (int c = minX; c < maxX; ++c)
        {
            Point pt;
            pt.x = c;
            pt.y = r;
            if (PointInTriangleGPU(pt, tri.points[0], tri.points[1], tri.points[2]))    
            { 
                int triImgIdx = (r * cols + c) * 3;
                device_tri_img[triImgIdx] = device_img[imgIdx]; 
                device_tri_img[triImgIdx + 1] = device_img[imgIdx + 1]; 
                device_tri_img[triImgIdx + 2] = device_img[imgIdx + 2]; 
            }   
        }                     
    }
}


void PrintDevice()
{
    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}


void fakeInit()
{
    cudaMalloc(&device_dummy, sizeof(uint8_t));   
    cudaFree(device_dummy);
}


void getGradGPU(cv::Mat &img)
{
    // PrintDevice();

    // double tolt_start = CycleTimer::currentSeconds();

    int rows = img.rows;
    int cols = img.cols;
    int numPixel = rows * cols;

    cv::Mat imgGray;
    imgGray.create(rows, cols, CV_8UC1);
    cv::cvtColor(img, imgGray, CV_BGR2GRAY);

    cudaMalloc(&device_img_gray, sizeof(uint8_t)*numPixel);    
    cudaMalloc(&device_grad, sizeof(float)*numPixel);
    cudaMemcpy(device_img_gray, imgGray.data, sizeof(uint8_t)*numPixel, cudaMemcpyHostToDevice);

    unsigned int n = 32;
    dim3 blockDim(n, n);
    dim3 gridDim((cols + n - 1) / n, (rows + n - 1) / n);

    // double comp_start = CycleTimer::currentSeconds();

    get_grad_kernel<<<gridDim, blockDim>>>(device_img_gray, device_grad, rows, cols);
    gpuErrchk(cudaDeviceSynchronize());

    // cout<<"Get grad GPU computation time: "<< (CycleTimer::currentSeconds() - comp_start) * 1000 <<"ms"<<endl;
    // cout<<"Get grad GPU total time: "<< (CycleTimer::currentSeconds() - tolt_start) * 1000 <<"ms"<<endl;
}


void selectVerticesGPU(float edgeThresh, float edgeP, float nonEdgeP, float boundP, int rows, int cols)
{
    // double comp_start = CycleTimer::currentSeconds();

    int numPixel = rows * cols;

    unsigned int n = 32;
    dim3 blockDim(n, n);
    dim3 gridDim((cols + n - 1) / n, (rows + n - 1) / n);

    curandState *device_state;
    cudaMalloc(&device_state, sizeof(curandState)*n*n);

    cudaMalloc(&device_ownerMap, sizeof(Point)*numPixel);
    cudaMemset(device_ownerMap, -1, sizeof(Point)*numPixel);

    setup_rand_kernel<<<1, blockDim>>>(device_state); // this curand_init is slow
    gpuErrchk(cudaDeviceSynchronize());  
    select_vertex_kernel<<<gridDim, blockDim>>>(device_grad, device_ownerMap, device_state, edgeThresh, edgeP, nonEdgeP, boundP, rows, cols);
    gpuErrchk(cudaDeviceSynchronize());   

    cudaFree(device_grad);
    cudaFree(device_state);

    // cout<<"Select vertices GPU time: "<< (CycleTimer::currentSeconds() - comp_start) * 1000 <<"ms"<<endl;
}

cv::Mat drawTriangleGPU(cv::Mat& img)
{
    // double comp_start = CycleTimer::currentSeconds();

    int rows = img.rows;
    int cols = img.cols;
    int numPixel = rows * cols;

    cudaMalloc(&device_img, sizeof(uint8_t)*numPixel*3);    
    cudaMalloc(&device_tri_img, sizeof(uint8_t)*numPixel*3);
    cudaMemcpy(device_img, img.data, sizeof(uint8_t)*numPixel*3, cudaMemcpyHostToDevice);

    int gridDim = (num_triangles + 255) / 256;
    draw_triangle_kernel<<<gridDim, 256>>>(device_triangles, num_triangles, device_img, device_tri_img, rows, cols); 
    gpuErrchk(cudaDeviceSynchronize());  
    cv::Mat tri_img;
    tri_img.create(rows, cols, CV_8UC3);
    cudaMemcpy(tri_img.data, device_tri_img, sizeof(uint8_t)*numPixel*3, cudaMemcpyDeviceToHost);

    cudaFree(device_triangles);

    // cout<<"Draw triangle time: "<< (CycleTimer::currentSeconds() - comp_start) * 1000 <<"ms"<<endl;

    return tri_img;
}

void DelauneyGPU(int rows, int cols)
{
    // define grid and block size
    unsigned int n = 32;
    dim3 blockDim(n, n);
    dim3 gridDim((cols + n - 1) / n, (rows + n - 1) / n);

    // double comp_start = CycleTimer::currentSeconds();

    // Step1 : find Voronoi graph
    int start_stepsize = NextPower2_CPU(min(rows, cols)) / 2;
    for(int stepsize = start_stepsize; stepsize>=1; stepsize /= 2)
    {
        voronoi_kernel<<<gridDim, blockDim>>>(device_ownerMap, stepsize, rows, cols);
        gpuErrchk(cudaDeviceSynchronize());
    }

    // Step 2: find the number of triangles
    int* device_triangle_cnts;
    cudaMalloc(&device_triangle_cnts, sizeof(int) * rows * cols);

    count_triangle_kernel<<<gridDim, blockDim>>>(device_ownerMap, rows, cols, device_triangle_cnts);
    gpuErrchk(cudaDeviceSynchronize());

    // Step 3: prefix sum #triangles
    int* device_sum_triangles;
    cudaMalloc(&device_sum_triangles, sizeof(int) * rows * cols);
    thrust::inclusive_scan(thrust::device, device_triangle_cnts, device_triangle_cnts + rows*cols, device_sum_triangles);

    // Step 4: build the triangles
    cudaMemcpy(&num_triangles, &device_sum_triangles[rows*cols-1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMalloc(&device_triangles, sizeof(Triangle) * num_triangles);
    triangle_kernel<<<gridDim, blockDim>>>(device_ownerMap, device_triangles, rows, cols, device_sum_triangles);
    gpuErrchk(cudaDeviceSynchronize());

    // copy triangle data to CPU
    // Triangle triangles[num_triangles];
    // cudaMemcpy(triangles, device_triangles, sizeof(Triangle)*num_triangles, cudaMemcpyDeviceToHost);

    // vector<Triangle> ret(triangles, triangles + num_triangles);

    // free
    cudaFree(device_triangle_cnts);
    cudaFree(device_sum_triangles);

    // cout<<"Delauney Core computation time: "<< (CycleTimer::currentSeconds() - comp_start) * 1000 <<"ms"<<endl;
}
