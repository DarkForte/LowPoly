#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <vector>
#include "delauney.h"
using namespace std;


vector<Triangle> DelauneyGPU(vector<Point> &seeds, vector<int> &owner, int rows, int cols)
{
    //int start_stepsize = NextPower2(min(rows, cols)) / 2;

    const Point dir[] = {Point(0,1), Point(0, -1), Point(1, 0), Point(-1, 0),
                         Point(1,1), Point(1, -1), Point(-1, 1), Point(-1,-1)};
}
