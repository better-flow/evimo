# Download and extract raw recordings
The drive should have at least 4TB of free space if the entire dataset is to be generated with default settings

Decompressing more than one file at a time is much faster.

The result should look like this:
```
>> ls -1 /media/$USER/EVIMO/raw
imo/
imo_ll/
raw_imo_ll.tar.gz
raw_imo.tar.gz
raw_sanity_ll.tar.gz
raw_sanity.tar.gz
raw_sfm_ll.tar.gz
raw_sfm.tar.gz
sanity/
sanity_ll/
sfm/
sfm_ll/
```

# Generate the dataset
## Swapfile (if necessary)
Generation currently requires about 80GB of memory (RAM). A large swapfile on an SSD will work fine.

These instructions may help with creating swapfile: https://askubuntu.com/a/1075516

## Generate a particular sequence
Generating a single sequence can take a few minutes to an hour depending on the seqeuence. More cores will make it faster.

```
cd evimo/tools/evimo2_docker
./docker_build.sh
./docker_run.sh /media/$USER/EVIMO
cd ~/catkin_ws; catkin_make; pip3 install -e ~/pydvs/lib; cd
source ~/catkin_ws/devel/setup.bash
cd ~/tools/evimo2_generate; ./generate.sh ~/EVIMO/raw/imo/eval/scene13_dyn_test_00
```

## Generate everything
Generating the entire dataset can take days.

```
cd evimo/tools/evimo2_docker
./docker_build.sh
./docker_run.sh /media/$USER/EVIMO
cd ~/catkin_ws; catkin_make; pip3 install -e ~/pydvs/lib; cd
source ~/catkin_ws/devel/setup.bash
cd ~/tools/evimo2_generate; ./generate_all.sh ~/EVIMO/raw
```

See the detailed tools descriptions below for more information.

# Generation Tools
All generation tools are located in `evimo/tools/evimo2_generation`.

## Clear
Deletes all generated files but leaves the raw recordings

Clear all:
```
./clear_all.sh ~/EVIMO/raw
```

Clear a specific recording:
```
./clear.sh ~/EVIMO/raw/imo/eva/scene13_dyn_test_00
```

## Generate
Runs for each camera in a sequence folder
* the offline tool to generate txt format
* the evimo-gen python tool to generat npz format and visualization frames
* ffmpeg to make the visualization video
* cleans up all intermediate artifacts to save TB's of disk space as it goes along
* all final artifacts are left in each sequences folder, they will be moved into the final dataset file/folder structure later

Generate all:
```
./generate_all.sh ~/raw
```

Generate a specific recording:
```
./generate.sh ~/EVIMO/raw/imo/eva/scene13_dyn_test_00
```

## Package

Checks that files that should have been made by `generate.sh` are present and copies or moves files into the released file/folder structure.

To do a dry run (check for missing generated files)

```
./package_all.py ~/EVIMO/raw ~/EVIMO/packaged dry
```

To do a real run (moves files and copies those that can't be moved):

Moving instead of copying saves over a 1 TB of drive space and makes the process fit on a 4TB drive.

```
./package_all.py ~/EVIMO/raw ~/EVIMO/packaged move
```

## Compress
Checks that files that should have been made by `generate.sh` are present and copies or moves files into the released file/folder structure.

To do a dry run:

```
./compress_packaged.py ~/EVIMO/packaged ~/EVIMO/compressed dry
```

To do a real run (moves files and copies those that can't be moved):

```
./compress_packaged.py ~/EVIMO/packaged ~/EVIMO/compressed compress
```
