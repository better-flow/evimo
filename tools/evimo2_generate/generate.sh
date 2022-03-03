#!/bin/bash


# Run: bash generate.sh <dataset_folder>

# INSTALL:
# 1) https://github.com/better-flow/evimo/wiki/Evimo-Pipeline-Setup
# 2) install https://github.com/better-flow/pydvs
# 3) Wiki: https://github.com/better-flow/evimo/wiki/Dataset-Configuration-Folder and https://github.com/better-flow/evimo/wiki/Ground-Truth-Format


PYDVS_DIR=~/pydvs # see https://github.com/better-flow/pydvs

npy_folder_to_npz_delete_npy() {
    zip -rjq $1 $2
    rm -rf $2
}

if [[ $# -eq 0 ]] ; then
    echo "No folder specified, exiting"
    exit 1
fi

if [[ $# -eq 1 ]] ; then
    echo "Folder: $1"
    wait_for_zip=true
fi

if [[ $# -eq 2 ]] ; then
    echo "Folder: $1"
    echo "There are two arguments, interpreting existence of the second to mean not to wait for the zip job to complete"
    wait_for_zip=false
fi


declare -a CameraArray=("flea3_7" "samsung_mono" "left_camera" "right_camera")
declare -a zip_pids=()
for cam in ${CameraArray[@]}; do
    echo $cam

    rm -rf $1/$cam/ground_truth

    S_NAME=$(basename $1)
    VIS_FOLDER=$1/$cam/ground_truth/vis
    VIDEO_DST=$1/${S_NAME}_${cam}.mp4
    rm $VIDEO_DST

    roslaunch evimo event_imo_offline.launch show:=-1 folder:=$1 camera_name:=$cam \
                                             generate:=true \
                                             save_3d:=false \
                                             fps:=60 \
                                             t_offset:=0 \
                                             t_len:=-1

    python3 $PYDVS_DIR/samples/evimo-gen.py --base_dir $1/$cam/ground_truth --skip_slice_vis --evimo2_npz --evimo2_no_compress

    # Compress npy to npz seperate process to save time several hours when doing a full generation of the dataset

    npy_folder_to_npz_delete_npy $1/$cam"/ground_truth/dataset_depth.npz" $1/$cam"/ground_truth/depth_npy"&
    zip_pids+=($!)

    npy_folder_to_npz_delete_npy $1/$cam"/ground_truth/dataset_mask.npz" $1/$cam"/ground_truth/mask_npy"&
    zip_pids+=($!)

    if [ -d $1"/classical_npy" ] 
    then
        echo "Not compressing dataset_classical, it doesn't exist"
    else
        npy_folder_to_npz_delete_npy $1/$cam"/ground_truth/dataset_classical.npz" $1/$cam"/ground_truth/classical_npy"&
        zip_pids+=($!)
    fi

    ffmpeg -r 60 -i $VIS_FOLDER/frame_%10d.png -c:v libx264 -vf \
        "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p \
        $VIDEO_DST
    rm -rf $VIS_FOLDER
done

# wait for all zip pids
if [ "$wait_for_zip" = true ]
then
    for pid in ${zip_pids[@]}; do
        if [ -f /proc/${pid} ]
        then
            echo "Waiting for npy_folder_to_npz_delete_npy pid: "$pid
            wait $pid
        fi
    done
else
    for pid in ${zip_pids[@]}; do
        if [ -f /proc/${pid}/cmdline ]
        then
            echo "npy_folder_to_npz_delete_npy pid: "$pid "is still running"
        fi
    done
fi
