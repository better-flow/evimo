# Noetic on Ubuntu 20.04 (focal) base image
from ros:noetic-ros-base-focal

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-pcl-conversions \
    ros-noetic-pcl-ros \
    ros-noetic-tf-conversions \
    ros-noetic-laser-geometry \
    ros-noetic-camera-info-manager \
    libyaml-cpp-dev \
    libzip-dev \
    git \
    python3-pip \
    python3-tk \
    ffmpeg \
    zip \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install matplotlib pyquaternion scipy tqdm pandas numpy --upgrade

RUN git clone https://github.com/aftersomemath/pydvs --branch evimo2_regen && \
    cd pydvs/lib && \
    pip3 install .
