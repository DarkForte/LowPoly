#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <ctime>
#include "triangle.h"
#include <vector>
#include <stdio.h>
#include <iostream>
#include "delauney.h"

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
using namespace std;

uint8_t* device_img = NULL;
float* device_grad = NULL;
Point* device_seeds = NULL;
Triangle* device_triangles = NULL;
int* device_owner = NULL;
int* mutex;
int* numTriangles;

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

__global__ void try_kernel()
{
    Triangle triangle_1;
    Triangle triangle(Point(0, 0), Point(0, 3), Point(3, 3));
    Point center = triangle.center();
    printf("%d, %d\n", center.x, center.y);
}

__global__ void voronoi_kernel(Point* device_seeds, int* device_owner, int stepsize, int rows, int cols)
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

        if(device_owner[Index(now_looking, cols)] == -1)
            continue;

        int cand_dist = dist(device_seeds[device_owner[Index(now_looking, cols)]], now_point);
        int now_owner = device_owner[Index(now_point, cols)];

        if(now_owner == -1 || cand_dist < dist(device_seeds[now_owner], now_point))
            device_owner[Index(now_point, cols)] = device_owner[Index(now_looking, cols)];
    }
}

__device__ void count_colors(Point now_point, int device_owner[], int colors[4], int &numColors, int cols)
{
    Point neighbor_dir[] = {Point(1, 0), Point(0, 1), Point(1, 1)};

    colors[0] = device_owner[Index(now_point, cols)];
    numColors = 1;
    for(Point now_dir: neighbor_dir)
    {
        Point next_point = now_point + now_dir;
        int newColor = device_owner[Index(next_point, cols)];
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
__global__ void count_triangle_kernel(int* device_owner, int rows, int cols, int* triangle_count)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= cols-1 || r >= rows-1)
        return;

    Point now_point(c, r);

    int colors[4] = {-1, -1, -1, -1};
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
__global__ void triangle_kernel(Point* device_seeds, int* device_owner, Triangle* device_triangles, int rows, int cols, int* device_sum_triangles)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= cols-1 || r >= rows-1)
        return;

    Point now_point(c, r);

    int colors[4] = {-1, -1, -1, -1};
    int numColors;
    count_colors(now_point, device_owner, colors, numColors, cols);

    int index = Index(now_point, cols);
    int prev_triangle_cnt = (index == 0)? 0: device_sum_triangles[index-1];

    if(numColors == 3)
    {
        Triangle triangle;
        for(int i=0; i<3; i++)
        {
            triangle.points[i] = device_seeds[colors[i]];
        }

        device_triangles[prev_triangle_cnt] = triangle;
    }
    else if(numColors == 4)
    {
        Triangle triangle1(device_seeds[device_owner[Index(now_point, cols)]],
                           device_seeds[device_owner[Index(now_point + Point(1, 0), cols)]],
                           device_seeds[device_owner[Index(now_point + Point(0, 1), cols)]]);

        Triangle triangle2(device_seeds[device_owner[Index(now_point + Point(1, 0), cols)]],
                           device_seeds[device_owner[Index(now_point + Point(0, 1), cols)]],
                           device_seeds[device_owner[Index(now_point + Point(1, 1), cols)]]);

        device_triangles[prev_triangle_cnt] = triangle1;
        device_triangles[prev_triangle_cnt+1] = triangle2;
    }
    return;
}


__global__ void get_grad_kernel(uint8_t* device_img, float* device_grad, int rows, int cols)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= cols || r >= rows)
        return;

    if (r > 0 && c > 0 && r < rows - 1 && c < cols - 1) // inside the image
    {
        float grad_x = -(float)device_img[(r - 1) * cols + c - 1] + (float)device_img[(r - 1) * cols + c + 1]
                       - 2 * (float)device_img[r * cols + c - 1]  + 2 * (float)device_img[r * cols + c + 1]
                       -(float)device_img[(r + 1) * cols + c - 1] + (float)device_img[(r + 1) * cols + c + 1];
        grad_x = abs(grad_x);

        float grad_y = -(float)device_img[(r - 1) * cols + c - 1] - 2 * (float)device_img[(r - 1) * cols + c] - (float)device_img[(r - 1) * cols + c + 1]
                       +(float)device_img[(r + 1) * cols + c - 1] + 2 * (float)device_img[(r + 1) * cols + c] + (float)device_img[(r + 1) * cols + c + 1];
        grad_y = abs(grad_y);

        float grad = grad_x / 2.0 + grad_y / 2.0;
        device_grad[r * cols + c] = grad;
    }

    else // set the boundary values to 0
    {
        device_grad[r * cols + c] = 0;
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


cv::Mat getGradGPU(cv::Mat &img)
{
    int rows = img.rows;
    int cols = img.cols;
    int numPixel = rows * cols;

    cv::Mat imgGray;
    imgGray.create(rows, cols, CV_8UC1);
    cv::cvtColor(img, imgGray, CV_BGR2GRAY);

    cudaMalloc(&device_img, sizeof(uint8_t)*numPixel);
    cudaMalloc(&device_grad, sizeof(float)*numPixel);
    cudaMemcpy(device_img, imgGray.data, sizeof(uint8_t)*numPixel, cudaMemcpyHostToDevice);

    unsigned int n = 32;
    dim3 blockDim(n, n);
    dim3 gridDim((cols + n - 1) / n, (rows + n - 1) / n);
    get_grad_kernel<<<gridDim, blockDim>>>(device_img, device_grad, rows, cols);
    gpuErrchk(cudaDeviceSynchronize());

    cv::Mat grad;
    grad.create(rows, cols, CV_32FC1);
    cudaMemcpy(grad.data, device_grad, sizeof(float)*numPixel, cudaMemcpyDeviceToHost);

    cudaFree(device_grad);
    
    return grad;
}


vector<Triangle> DelauneyGPU(Point* seeds, int numSeeds, int* owner, int rows, int cols)
{
    PrintDevice();

    // put seeds on the graph
    for(int i=0; i<numSeeds; i++)
    {
        Point seed = seeds[i];
        owner[Index_CPU(seed, cols)] = i;
    }

    // define grid and block size
    unsigned int n = 32;
    dim3 blockDim(n, n);
    dim3 gridDim((cols + n - 1) / n, (rows + n - 1) / n);


    // transfer seeds and owner to device
    clock_t mem_start = clock();

    cudaMalloc(&device_seeds, sizeof(Point)*numSeeds);
    cudaMalloc(&device_owner, sizeof(int)*rows*cols);
    printf("Mem Time: %lf\n", (clock() - mem_start) / (double)(CLOCKS_PER_SEC / 1000));

    cudaMemcpy(device_seeds, seeds, sizeof(Point)*numSeeds, cudaMemcpyHostToDevice);


    cudaMemcpy(device_owner, owner, sizeof(int)*rows*cols, cudaMemcpyHostToDevice);

    clock_t comp_start = clock();

    // Step1 : find Voronoi graph
    int start_stepsize = NextPower2_CPU(min(rows, cols)) / 2;
    for(int stepsize = start_stepsize; stepsize>=1; stepsize /= 2)
    {
        voronoi_kernel<<<gridDim, blockDim>>>(device_seeds, device_owner, stepsize, rows, cols);
        gpuErrchk(cudaDeviceSynchronize());
    }

    // copy owner data to CPU
    cudaMemcpy(owner, device_owner, sizeof(int)*rows*cols, cudaMemcpyDeviceToHost);

    // Step 2: find the number of triangles
    int* device_triangle_cnts;
    cudaMalloc(&device_triangle_cnts, sizeof(int) * rows * cols);

    count_triangle_kernel<<<gridDim, blockDim>>>(device_owner, rows, cols, device_triangle_cnts);
    gpuErrchk(cudaDeviceSynchronize());

    // Step 3: prefix sum #triangles
    int* device_sum_triangles;
    cudaMalloc(&device_sum_triangles, sizeof(int) * rows * cols);
    thrust::inclusive_scan(thrust::device, device_triangle_cnts, device_triangle_cnts + rows*cols, device_sum_triangles);

    // Step 4: build the triangles
    Triangle* device_triangles;
    int num_triangles;
    cudaMemcpy(&num_triangles, &device_sum_triangles[rows*cols-1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMalloc(&device_triangles, sizeof(Triangle) * num_triangles);
    triangle_kernel<<<gridDim, blockDim>>>(device_seeds, device_owner, device_triangles, rows, cols, device_sum_triangles);
    gpuErrchk(cudaDeviceSynchronize());

    // copy triangle data to CPU
    Triangle triangles[num_triangles];
    cudaMemcpy(triangles, device_triangles, sizeof(Triangle)*num_triangles, cudaMemcpyDeviceToHost);

    vector<Triangle> ret(triangles, triangles + num_triangles);

    // free
    cudaFree(device_seeds);
    cudaFree(device_owner);
    cudaFree(device_triangle_cnts);
    cudaFree(device_sum_triangles);
    cudaFree(device_triangles);

    cout<<"Core computation time: "<< (clock() - comp_start) / (double)(CLOCKS_PER_SEC / 1000) <<endl;

    return ret;
}
