cmake_minimum_required(VERSION 2.8.3)
project(event_imo_datagen)


set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE Release)

add_definitions(-std=c++14)



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


find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(OpenCV REQUIRED)


catkin_package(CATKIN_DEPENDS message_runtime sensor_msgs std_msgs geometry_msgs vicon)


include_directories(${catkin_INCLUDE_DIRS}
	                ${PCL_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})




###########
## Build ##
###########
add_executable(event_imo_online online.cpp)

target_link_libraries(event_imo_online
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)


add_executable(event_imo_offline offline.cpp)

target_link_libraries(event_imo_offline
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
)
