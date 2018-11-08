# CMakeList.txt : CMake project for LowPoly, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 2.8)

# Add source to this project's executable.
file(GLOB LowPoly_src
    "*.h"
    "*.cpp"
)

add_executable(LowPoly ${LowPoly_src})

# TODO: Add tests and install targets if needed.
