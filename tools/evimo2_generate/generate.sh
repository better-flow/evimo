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
for cam in ${CameraArray[@]}; do
    echo $cam

    rm -rf $1/$cam/ground_truth

    roslaunch evimo event_imo_offline.launch show:=-1 folder:=$1 camera_name:=$cam \
                                             generate:=true \
                                             save_3d:=false \
                                             fps:=60 \
                                             t_offset:=0 \
                                             t_len:=-1
done

# This script splits the output of the offline tool using information from all the cameras
python3 split_offline_txt_output.py --base_dir $1


declare -a zip_pids=()
for cam in ${CameraArray[@]}; do
    # Get a sorted list of ground truth split folders
    gt_folders=$(find "$1/$cam" -mindepth 1 -maxdepth 1 -type d -name "ground_truth_*")
    gt_folders=$(echo $gt_folders | xargs -n1 | sort | xargs)
    gt_folders=($gt_folders)


    for gt_folder in "${gt_folders[@]}"; do
        echo $gt_folder
        python3 $PYDVS_DIR/samples/evimo-gen.py --base_dir $gt_folder --skip_slice_vis --evimo2_npz --evimo2_no_compress

        # Compress npy to npz seperate process to save time when doing a full generation of the dataset
        npy_folder_to_npz_delete_npy $gt_folder"/dataset_depth.npz" $gt_folder"/depth_npy"&
        zip_pids+=($!)

        npy_folder_to_npz_delete_npy $gt_folder"/dataset_mask.npz" $gt_folder"/mask_npy"&
        zip_pids+=($!)

        npy_folder_to_npz_delete_npy $gt_folder"/dataset_classical.npz" $gt_folder"/classical_npy"&
        zip_pids+=($!)


        # Generate video
        S_NAME=$(basename $1)
        GT_NAME=$(basename $gt_folder)
        VIS_FOLDER=$gt_folder/vis
        VIDEO_DST=$1/${S_NAME}_${cam}_${GT_NAME}.mp4
        rm $VIDEO_DST

        if [ $cam = "flea3_7" ]
        then
            fps=30
        else
            fps=60
        fi

        ffmpeg -r $fps -i $VIS_FOLDER/frame_%10d.png -c:v libx264 -vf \
            "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p \
            $VIDEO_DST
        rm -rf $VIS_FOLDER
    done

    # To save disk space
    rm -rf $1/$cam/ground_truth
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
