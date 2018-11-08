# CMakeList.txt : CMake project for LowPoly, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 2.8)
set(EXECUTABLE_OUTPUT_PATH ../)
LINK_DIRECTORIES(~/softwares/opencv-3.4.1/install/lib64)

# Add source to this project's executable.
file(GLOB LowPoly_src
    "*.h"
    "*.cpp"
)

add_executable(LowPoly ${LowPoly_src})
TARGET_LINK_LIBRARIES(LowPoly opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc)
#target_link_libraries( LowPoly ${OpenCV_LIBS} )

# TODO: Add tests and install targets if needed.
