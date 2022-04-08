#!/bin/bash

# Usage:
# ./download_evimo2_v2.sh npz path/to/output_directory
# or
# ./download_evimo2_v2.sh txt path/to/output_directory

if [ "$#" -ne 2 ]; then
    echo "There must be two parameters, see README.md"
else
    FORMAT=$1
    OUTPUT_DIR=$2

    DOWNLOAD_URL="https://obj.umiacs.umd.edu/evimo2v2"$FORMAT"/"

    # Can comment these out in order to skip downloading some files
    files=(
    flea3_7_imo.tar.gz
    flea3_7_sanity.tar.gz
    flea3_7_sanity_ll.tar.gz
    flea3_7_sfm.tar.gz
    flea3_7_sfm_ll.tar.gz
    left_camera_imo.tar.gz
    left_camera_imo_ll.tar.gz
    left_camera_sanity.tar.gz
    left_camera_sanity_ll.tar.gz
    left_camera_sfm.tar.gz
    left_camera_sfm_ll.tar.gz
    right_camera_imo.tar.gz
    right_camera_imo_ll.tar.gz
    right_camera_sanity.tar.gz
    right_camera_sanity_ll.tar.gz
    right_camera_sfm.tar.gz
    right_camera_sfm_ll.tar.gz
    samsung_mono_imo.tar.gz
    samsung_mono_imo_ll.tar.gz
    samsung_mono_sanity.tar.gz
    samsung_mono_sanity_ll.tar.gz
    samsung_mono_sfm.tar.gz
    samsung_mono_sfm_ll.tar.gz
    )

    for file in "${files[@]}"
    do
        full_file_name=$FORMAT"_"$file
        echo $full_file_name
        file_url=$DOWNLOAD_URL$full_file_name
        echo $file_url

        cd $OUTPUT_DIR
        wget -O - $file_url | tar -C $OUTPUT_DIR -zxf -
    done
fi