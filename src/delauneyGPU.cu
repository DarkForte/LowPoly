#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <vector>
#include <stdio.h>
#include <iostream>
#include "delauney.h"
using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void tryKernel()
{
    Triangle triangle(Point(0, 0), Point(0, 3), Point(3, 3));
    Point center = triangle.center();
    printf("%d, %d\n", center.x, center.y);
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

vector<Triangle> DelauneyGPU(vector<Point> &seeds, vector<int> &owner, int rows, int cols)
{
    PrintDevice();

    //int start_stepsize = NextPower2(min(rows, cols)) / 2;

    const Point dir[] = {Point(0,1), Point(0, -1), Point(1, 0), Point(-1, 0),
                         Point(1,1), Point(1, -1), Point(-1, 1), Point(-1,-1)};

    tryKernel<<<1,1>>>();
    gpuErrchk(cudaDeviceSynchronize());

    return vector<Triangle>();
}
