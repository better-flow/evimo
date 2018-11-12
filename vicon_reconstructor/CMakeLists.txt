cmake_minimum_required(VERSION 2.8.3)
project(vicon_reconstructor)


set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE Release)

#set(CMAKE_BUILD_TYPE Debug)
add_definitions(-std=c++14)


find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  vicon
)



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
)

find_package(OpenCV REQUIRED)


catkin_package(CATKIN_DEPENDS message_runtime sensor_msgs std_msgs geometry_msgs vicon)




include_directories(${catkin_INCLUDE_DIRS}
	                ${PCL_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})




###########
## Build ##
###########
add_executable(vrec vrec.cpp)
target_link_libraries(vrec
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)


add_executable(vrec_photo vrec_photo.cpp)
target_link_libraries(vrec_photo
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)
