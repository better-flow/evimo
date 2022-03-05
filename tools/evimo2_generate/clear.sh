#!/bin/bash

echo "removing generated files from folder: $1"

declare -a CameraArray=("flea3_7" "samsung_mono" "left_camera" "right_camera")
for cam in ${CameraArray[@]}; do
    rm -rf $1/$cam/ground_truth*

    S_NAME=$(basename $1)
    VIS_FOLDER=$1/$cam/ground_truth/vis
    VIDEO_DST=$1/${S_NAME}_${cam}.mp4
    rm $1/${S_NAME}_${cam}*.mp4
done

rm "$1/frame_time_plot.pdf"
