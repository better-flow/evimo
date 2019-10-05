cmake_minimum_required(VERSION 2.8.3)
project(evimo)

#set(CMAKE_C_COMPILER "/usr/bin/clang-6.0")
#set(CMAKE_CXX_COMPILER "/usr/bin/clang++-6.0")

#set(CMAKE_CXX_FLAGS_RELEASE "-O3")
#set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED COMPONENTS
             cv_bridge
             geometry_msgs
             image_transport
             message_generation
             pcl_conversions
             pcl_ros
             roscpp
             sensor_msgs
             visualization_msgs
             tf
             tf_conversions
             laser_geometry
             std_msgs
             vicon
             dvs_msgs
)

find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(OpenCV REQUIRED)

catkin_package(CATKIN_DEPENDS message_runtime sensor_msgs std_msgs geometry_msgs vicon dvs_msgs)

include_directories(.
                    ${catkin_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

###########
## Build ##
###########
add_executable(datagen_online online.cpp)

target_link_libraries(datagen_online
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)


add_executable(datagen_offline annotation_backprojector.h
                               offline.cpp
                               dataset.cpp
                               dataset_frame.cpp)

target_link_libraries(datagen_offline
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
)
