# EVIMO Pipeline Setup on Host OS

A [Docker container](docker-environment.md) is provided which can be used for most tasks with the dataset. However, instructions for setting up a ROS workspace on the host OS are provided below.

1. The code was tested on Ubuntu 18.04 and 20.04 and consists of multiple tools for visualization, calibration and ground truth generation.

2. While the released dataset is self-contained, in order to generate ground truth from raw `.bag` recordings you will need ROS installed on your system:
  - [ROS Noetic installation](http://wiki.ros.org/noetic/Installation/Ubuntu "Read this to install ROS on your system")
  - [ROS Tutorials](http://wiki.ros.org/ROS/Tutorials "This is a set of brief ROS tutorials")
  - Overall setup is similar to our [better-flow](https://github.com/better-flow/better-flow) project; you might be able to follow the instructions from the project page.

## Setup
1. Make sure ROS is [installed](http://wiki.ros.org/noetic/Installation/Ubuntu); This project also relies on [OpenCV](https://opencv.org/) and (to a lesser degree) [PCL](https://pointclouds.org/).
2. If you do not have a catkin workspace set up:
  1. Download the *cognifli* code (see [project page](https://github.com/ncos/cognifli) for more details):
  ```bash
  cd ~/
  git clone https://github.com/ncos/cognifli
  ```
  2. Run the *INSTALL.py* configuration tool to set up your catkin workspace:
  ```bash
  cd ~/cognifli/contrib
  ./INSTALL.py
  ```
3. Download packages with DVS/Vicon ROS message descriptions: [link](https://drive.google.com/file/d/15oSCxfUN8oAskz-hLeDGBZoq5HLyYYxM/view?usp=sharing) 
4. **OR** Download [DVS](https://github.com/uzh-rpg/rpg_dvs_ros) and [Vicon](https://github.com/KumarRobotics/vicon) packages from their repositories:
```bash
cd ~/cognifli/src
git clone https://github.com/catkin/catkin_simple.git
git clone https://github.com/uzh-rpg/rpg_dvs_ros.git
git clone https://github.com/KumarRobotics/vicon
```
**Note**: only message description parts of the packages are needed; you can remove the rest of the package in case you get compilation errors.

5. Download Evimo [source code](https://github.com/better-flow/evimo):
```bash
cd ~/cognifli/src
git clone https://github.com/better-flow/evimo
```

6. Build the project:
```bash
cd ~/cognifli
catkin_make
```

**Note**: We use [DVS](https://github.com/uzh-rpg/rpg_dvs_ros) event message format for both Samsung and Prophesee recordings.
