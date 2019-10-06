cmake_minimum_required(VERSION 2.8.3)
project(better_flow)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g")
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)


set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")

find_package(OpenCV REQUIRED)

###########
## Build ##
###########

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(rosbag REQUIRED)
find_package(dvs_msgs REQUIRED)

include_directories(
    ${rosbag_INCLUDE_DIRS}
    ${dvs_msgs_INCLUDE_DIRS}
)

add_executable(img_extract
    img_extract.cpp
)

target_link_libraries(img_extract
    ${rosbag_LIBRARIES}
    ${OpenCV_LIBS}
)
