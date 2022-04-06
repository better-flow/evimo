#!/bin/bash

DATA_ROOT=$1

# Get a sorted list of sequence folders
folders=$(find $DATA_ROOT -mindepth 3 -maxdepth 3 -type d)
folders=$(echo $folders | xargs -n1 | sort | xargs)
folders=($folders)

# Optionally use this hack to run just a few
# folders=(
# ~/EVIMO/raw/imo/eval/scene13_dyn_test_00
# ~/EVIMO/raw/imo/eval/scene13_dyn_test_05
# )

for folder in "${folders[@]}"
do
    sequence_folder=$folder
    bash ./generate.sh $sequence_folder no_wait_for_zip
done

echo "Checking for zip jobs that may still be running. Found:"
zip_pids_search=$(pgrep -f "[z]ip -rjq")
for pid in ${zip_pids_search[@]}; do
    cat /proc/${pid}/cmdline | xargs -0 echo
done
