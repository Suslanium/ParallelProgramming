cmake_minimum_required(VERSION 3.23)
project(ParallelProgramming)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp")

find_library(LODEPNG_TO_INCLUDE lodepng ${CMAKE_SOURCE_DIR}/lib/lodepng/lib)
find_path(LODEPNG_INCLUDES lodepng.h ${CMAKE_SOURCE_DIR}/lib/lodepng/include)
find_package(OpenCL REQUIRED)
include_directories(${LODEPNG_INCLUDES})
include_directories(${OpenCL_INCLUDE_DIRS})

add_executable(ParallelProgramming main.cpp)
target_link_libraries(ParallelProgramming ${LODEPNG_TO_INCLUDE})
target_link_libraries(ParallelProgramming ${OpenCL_LIBRARIES})