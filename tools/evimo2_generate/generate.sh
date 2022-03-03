#!/bin/bash


# Run: bash generate.sh <dataset_folder>

# INSTALL:
# 1) https://github.com/better-flow/evimo/wiki/Evimo-Pipeline-Setup
# 2) install https://github.com/better-flow/pydvs
# 3) Wiki: https://github.com/better-flow/evimo/wiki/Dataset-Configuration-Folder and https://github.com/better-flow/evimo/wiki/Ground-Truth-Format


PYDVS_DIR=~/pydvs # see https://github.com/better-flow/pydvs



echo "Folder: $1"

declare -a CameraArray=("flea3_7" "samsung_mono" "left_camera" "right_camera")
for cam in ${CameraArray[@]}; do
    echo $cam

    S_NAME=$(basename $1)
    VIS_FOLDER=$1/$cam/ground_truth/vis
    VIDEO_DST=$1/${S_NAME}_${cam}.mp4
    rm -rf $1/$cam/ground_truth
    rm $VIDEO_DST

    roslaunch evimo event_imo_offline.launch show:=-1 folder:=$1 camera_name:=$cam \
                                             generate:=true \
                                             save_3d:=false \
                                             fps:=60 \
                                             t_offset:=0 \
                                             t_len:=-1

    python3 $PYDVS_DIR/samples/evimo-gen.py --base_dir $1/$cam/ground_truth

    ffmpeg -r 60 -i $VIS_FOLDER/frame_%10d.png -c:v libx264 -vf \
        "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p \
        $VIDEO_DST
done
