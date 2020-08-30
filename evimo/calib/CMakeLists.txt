cmake_minimum_required(VERSION 2.8)
project(WandDetectorTest)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
add_executable(detect_wand detect_wand.cpp )
target_link_libraries(detect_wand ${OpenCV_LIBS} )
