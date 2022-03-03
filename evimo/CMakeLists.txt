cmake_minimum_required(VERSION 3.12)
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
find_package(ZLIB REQUIRED)



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


SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cnpy/cmake" ${CMAKE_MODULE_PATH})
FIND_PACKAGE(LibZip REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})
add_library(cnpy SHARED "cnpy/cnpy.cpp")
target_include_directories(cnpy PUBLIC ${LIBZIP_INCLUDE_DIR_ZIP})
target_link_libraries(cnpy ${ZLIB_LIBRARIES} ${LIBZIP_LIBRARIES})

add_executable(datagen_online plot.cpp
                              online.cpp
                              dataset.cpp
                              dataset_frame.cpp
)
add_dependencies(datagen_online ${catkin_EXPORTED_TARGETS})
target_link_libraries(datagen_online
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)


add_executable(datagen_offline annotation_backprojector.h
                               plot.cpp
                               offline.cpp
                               dataset.cpp
                               dataset_frame.cpp)
add_dependencies(datagen_offline ${catkin_EXPORTED_TARGETS})
target_link_libraries(datagen_offline
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
    cnpy
)


add_executable(bag_sanity bag_sanity/bag_sanity.cpp)
add_dependencies(bag_sanity ${catkin_EXPORTED_TARGETS})
target_link_libraries(bag_sanity
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
)

add_executable(topic_sanity bag_sanity/topic_sanity.cpp)
add_dependencies(topic_sanity ${catkin_EXPORTED_TARGETS})
target_link_libraries(topic_sanity
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
)



find_package(Python3 COMPONENTS Development NumPy)
add_executable(refine calib/refine_calibration.cpp
                      dataset.cpp)
target_include_directories(refine PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
add_dependencies(refine ${catkin_EXPORTED_TARGETS})
target_link_libraries(refine
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    Python3::Python
    Python3::NumPy
)


find_package(X11 REQUIRED)
add_executable(collect calib/collect.cpp
                       dataset.cpp)
add_dependencies(collect ${catkin_EXPORTED_TARGETS})
target_link_libraries(collect
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
    ${X11_LIBRARIES}

)
