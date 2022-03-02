#!/bin/bash
set -xe

if [ -d "docker_home" ] 
then
    echo "docker_home already exists, skipping creation"
else
    mkdir -p docker_home

    # Source ROS files always
    echo "source catkin_ws/devel/setup.bash" >> docker_home/.bashrc
fi

if [ -d "catkin_ws" ] 
then
    echo "catkin_ws already exists, skipping creation"
else
    mkdir -p catkin_ws/src
    git clone git@github.com:KumarRobotics/vicon.git catkin_ws/src/vicon
    rm -rf catkin_ws/src/vicon/vicon_driver \
           catkin_ws/src/vicon/vicon_odom \
           catkin_ws/src/vicon/ipc
    cd catkin_ws/src/vicon
    git apply ../../../vicon_cmake_disable.patch
    cd ../../..

    git clone git@github.com:uzh-rpg/rpg_dvs_ros.git catkin_ws/src/rpg_dvs_ros
    rm -rf catkin_ws/src/rpg_dvs_ros/davis_ros_driver \
           catkin_ws/src/rpg_dvs_ros/dvs_calibration  \
           catkin_ws/src/rpg_dvs_ros/dvs_calibration_gui \
           catkin_ws/src/rpg_dvs_ros/dvs_file_writer  \
           catkin_ws/src/rpg_dvs_ros/dvs_renderer     \
           catkin_ws/src/rpg_dvs_ros/dvs_ros_driver   \
           catkin_ws/src/rpg_dvs_ros/dvxplorer_ros_driver
fi

docker build --tag ev_imo:1.0 .
