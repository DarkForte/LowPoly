# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 2.8)

project ("LowPoly")

FIND_PACKAGE(CUDA REQUIRED)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O0 -g --gpu-architecture compute_35 -std=c++11")


# Include sub-projects.
add_subdirectory ("src")
