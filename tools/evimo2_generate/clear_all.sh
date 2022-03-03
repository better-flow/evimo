#!/bin/bash

DATA_ROOT=$1

# Get a sorted list of sequence folders
folders=$(find $DATA_ROOT -mindepth 3 -maxdepth 3 -type d)
folders=$(echo $folders | xargs -n1 | sort | xargs)
folders=($folders)

# Optionally use this hack to run just a few
# folders=(
# ./imo/eval/scene13_dyn_test_00
# ./imo/eval/scene13_dyn_test_05
# ./imo/eval/scene14_dyn_test_03
# )

for folder in "${folders[@]}"
do
    sequence_folder=$DATA_ROOT/$folder
    bash ./clear.sh $sequence_folder
done
