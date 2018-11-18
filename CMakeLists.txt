# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 2.8)

project ("LowPoly")

FIND_PACKAGE(CUDA REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")
SET (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-O3;-gencode arch=compute_61,code=sm_61;--std=c++11;")

message(${CUDA_NVCC_FLAGS})

# Include sub-projects.
add_subdirectory ("src")
