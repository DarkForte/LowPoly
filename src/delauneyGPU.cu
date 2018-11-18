#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "triangle.h"
#include <vector>
#include <stdio.h>
#include <iostream>
#include "delauney.h"
using namespace std;

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

__device__ void lock(int *mutex) {
    while (atomicCAS(mutex, 0, 1) != 0);
}

__device__ void unlock(int *mutex) {
    atomicExch(mutex, 0);
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

__global__ void triangle_kernel(Point* device_seeds, int* device_owner, Triangle* device_triangles, int rows, int cols, int* numTriangles, int* mutex)
{   
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= cols-1 || r >= rows-1)
        return;

    Point neighbor_dir[] = {Point(1, 0), Point(0, 1), Point(1, 1)};

    Point now_point(c, r);

    int colors[4] = {-1, -1, -1, -1};
    colors[0] = device_owner[Index(now_point, cols)];
    int numColors = 1;
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

    if(numColors == 3)
    {
        Triangle triangle;
        int p = 0;
        for(int index: colors)
            triangle.points[p++] = device_seeds[index];

        lock(mutex);
        device_triangles[*numTriangles] = triangle;
        *numTriangles ++;
        unlock(mutex);
    }
    else if(numColors == 4)
    {
        Triangle triangle1(device_seeds[device_owner[Index(now_point, cols)]],
                           device_seeds[device_owner[Index(now_point + Point(1, 0), cols)]],
                           device_seeds[device_owner[Index(now_point + Point(0, 1), cols)]]);

        Triangle triangle2(device_seeds[device_owner[Index(now_point + Point(1, 0), cols)]],
                           device_seeds[device_owner[Index(now_point + Point(0, 1), cols)]],
                           device_seeds[device_owner[Index(now_point + Point(1, 1), cols)]]);
        lock(mutex);
        device_triangles[*numTriangles] = triangle1;
        *numTriangles ++;
        device_triangles[*numTriangles] = triangle2;
        *numTriangles ++;
        unlock(mutex);
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

Triangle* DelauneyGPU(Point* seeds, int numSeeds, int* owner, int rows, int cols)
{
    // PrintDevice();

    // initialize seeds
    for(int i=0; i<numSeeds; i++)
    {
        Point seed = seeds[i];
        owner[Index_CPU(seed, cols)] = i;
    }

    // define grid and block size
    int n = 32;
    dim3 blockDim(n, n);
    dim3 gridDim((cols + n - 1) / n, (rows + n - 1) / n);

    // transfer seeds and owner to device
    cudaMalloc(&device_seeds, sizeof(Point)*numSeeds);
    cudaMalloc(&device_owner, sizeof(int)*rows*cols);
    cudaMemcpy(device_seeds, seeds, sizeof(Point)*numSeeds, cudaMemcpyHostToDevice);
    cudaMemcpy(device_owner, owner, sizeof(int)*rows*cols, cudaMemcpyHostToDevice);

    // finding voronoi graph
    int start_stepsize = NextPower2_CPU(min(rows, cols)) / 2;
    for(int stepsize = start_stepsize; stepsize>=1; stepsize /= 2)
    {
        voronoi_kernel<<<gridDim, blockDim>>>(device_seeds, device_owner, stepsize, rows, cols);
        gpuErrchk(cudaDeviceSynchronize());
    }
    gpuErrchk(cudaDeviceSynchronize());

    // copy owner data to CPU
    cudaMemcpy(owner, device_owner, sizeof(int)*rows*cols, cudaMemcpyDeviceToHost);

    // // finding triangles
    // cudaMalloc(&device_triangles, sizeof(Triangle)*numSeeds*2);
    // cudaMalloc(&mutex, sizeof(int));
    // cudaMemset(mutex, 0, sizeof(int));
    // cudaMalloc(&numTriangles, sizeof(int));
    // cudaMemset(numTriangles, 0, sizeof(int));
    // triangle_kernel<<<gridDim, blockDim>>>(device_seeds, device_owner, device_triangles, rows, cols, numTriangles, mutex);
    // gpuErrchk(cudaDeviceSynchronize());

    // // copy triangle data to CPU
    // int* numTri;
    // cudaMemcpy(numTri, numTriangles, sizeof(int)*1, cudaMemcpyDeviceToHost);
    // Triangle triangles[*numTri];
    // cudaMemcpy(triangles, device_triangles, sizeof(Triangle)*(*numTri), cudaMemcpyDeviceToHost);

    // free
    cudaFree(device_seeds);
    cudaFree(device_owner);

    return NULL;
}
