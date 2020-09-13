cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(text2cloud)


#set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
#set(CMAKE_BUILD_TYPE Debug)

find_package(PCL REQUIRED)


include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (text_events2cloud text_events2cloud.cpp)
target_link_libraries (text_events2cloud ${PCL_LIBRARIES})

add_executable (text_scan2cloud text_scan2cloud.cpp)
target_link_libraries (text_scan2cloud ${PCL_LIBRARIES})
