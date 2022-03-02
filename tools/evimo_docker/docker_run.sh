#!/bin/bash
#
# The first argument is full path to evimo_data_config
# https://github.com/better-flow/evimo_data_config
# For example:
# ./docker_run.sh /media/$USER/EVIMO/evimo_data_config

docker run -it \
    --user=$(id -u $USER):$(id -g $USER) \
    --env="DISPLAY" \
    --workdir="/home/$USER" \
    --volume=$(pwd)"/docker_home:/home/$USER" \
    --volume=$(pwd)"/catkin_ws:/home/$USER/catkin_ws" \
    --volume=$(pwd)"/../../evimo:/home/$USER/catkin_ws/src/evimo" \
    --volume=$1":/home/$USER/evimo_data_config" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    ev_imo:1.0 \
    bash

