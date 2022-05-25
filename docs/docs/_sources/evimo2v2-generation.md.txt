# Generating EVIMO2 v2
## Preparation
### Setup Docker container
Use the instructions [here](docker-environment.md).

### Download
Download and extract the raw recordings from the [downloads page](https://better-flow.github.io/evimo/download_evimo_2.html)

The drive should have at least 4TB of free space if the entire dataset is to be generated with default settings.

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

### Swapfile (if necessary)
Generation currently requires about 80GB of memory (RAM). A large swapfile on an SSD will work.

[These instructions](https://askubuntu.com/a/1075516) may help with creating swapfile.

## Generate a particular sequence
Generating a single sequence can take a few minutes to an hour depending on the seqeuence. More cores will make it faster.

```bash
cd evimo/tools/evimo2_docker
./docker_build.sh
./docker_run.sh /media/$USER/EVIMO
cd ~/catkin_ws; catkin_make; pip3 install -e ~/pydvs/lib; cd
source ~/catkin_ws/devel/setup.bash
cd ~/tools/evimo2_generate; ./generate.sh ~/EVIMO/raw/imo/eval/scene13_dyn_test_00
```

## Generate everything
Generating the entire dataset can take over 24 hours.

```bash
cd evimo/tools/evimo2_docker
./docker_build.sh
./docker_run.sh /media/$USER/EVIMO
cd ~/tools/evimo2_generate
./generate_all.sh ~/EVIMO/raw
./package_all.py ~/EVIMO/raw ~/EVIMO/packaged move
./compress_packaged.py ~/EVIMO/packaged ~/EVIMO/compressed compress
```

See the detailed tools descriptions below for more information.


## Remove bad sequences from packaged output

After manual evaluation, the following sequences were found to contain no usable data and so were removed from the packaged output before compressing.

```
txt/flea3_7/imo/train/scene6_dyn_train_00_000000
txt/left_camera/imo/train/scene6_dyn_train_00_000000
txt/right_camera/imo/train/scene6_dyn_train_00_000000
txt/samsung_mono/imo/train/scene6_dyn_train_00_000000

txt/left_camera/imo_ll/eval/scene16_d_dyn_test_01_000000
txt/right_camera/imo_ll/eval/scene16_d_dyn_test_01_000000
txt/samsung_mono/imo_ll/eval/scene16_d_dyn_test_01_000000

txt/flea3_7/sanity/depth_var/depth_var_1_ud_000000
txt/left_camera/sanity/depth_var/depth_var_1_ud_000000
txt/right_camera/sanity/depth_var/depth_var_1_ud_000000
txt/samsung_mono/sanity/depth_var/depth_var_1_ud_000000

txt/flea3_7/sfm/train/scene7_03_000002
txt/left_camera/sfm/train/scene7_03_000002
txt/right_camera/sfm/train/scene7_03_000002
txt/samsung_mono/sfm/train/scene7_03_000002

txt/flea3_7/sfm/train/seq_1_5_000001
txt/left_camera/sfm/train/seq_1_5_000001
txt/right_camera/sfm/train/seq_1_5_000001

npz/flea3_7/imo/train/scene6_dyn_train_00_000000
npz/left_camera/imo/train/scene6_dyn_train_00_000000
npz/right_camera/imo/train/scene6_dyn_train_00_000000
npz/samsung_mono/imo/train/scene6_dyn_train_00_000000

npz/left_camera/imo_ll/eval/scene16_d_dyn_test_01_000000
npz/right_camera/imo_ll/eval/scene16_d_dyn_test_01_000000
npz/samsung_mono/imo_ll/eval/scene16_d_dyn_test_01_000000

npz/flea3_7/sanity/depth_var/depth_var_1_ud_000000
npz/left_camera/sanity/depth_var/depth_var_1_ud_000000
npz/right_camera/sanity/depth_var/depth_var_1_ud_000000
npz/samsung_mono/sanity/depth_var/depth_var_1_ud_000000

npz/flea3_7/sfm/train/scene7_03_000002
npz/left_camera/sfm/train/scene7_03_000002
npz/right_camera/sfm/train/scene7_03_000002
npz/samsung_mono/sfm/train/scene7_03_000002

npz/flea3_7/sfm/train/seq_1_5_000001
npz/left_camera/sfm/train/seq_1_5_000001
npz/right_camera/sfm/train/seq_1_5_000001

video/flea3_7/imo/train/scene6_dyn_train_00_flea3_7_ground_truth_000000.mp4
video/left_camera/imo/train/scene6_dyn_train_00_left_camera_ground_truth_000000.mp4
video/right_camera/imo/train/scene6_dyn_train_00_right_camera_ground_truth_000000.mp4
video/samsung_mono/imo/train/scene6_dyn_train_00_samsung_mono_ground_truth_000000.mp4

video/left_camera/imo_ll/eval/scene16_d_dyn_test_01_left_camera_ground_truth_000000.mp4
video/right_camera/imo_ll/eval/scene16_d_dyn_test_01_right_camera_ground_truth_000000.mp4
video/samsung_mono/imo_ll/eval/scene16_d_dyn_test_01_samsung_mono_ground_truth_000000.mp4

video/flea3_7/sanity/depth_var/depth_var_1_ud_flea3_7_ground_truth_000000.mp4
video/left_camera/sanity/depth_var/depth_var_1_ud_left_camera_ground_truth_000000.mp4
video/right_camera/sanity/depth_var/depth_var_1_ud_right_camera_ground_truth_000000.mp4
video/samsung_mono/sanity/depth_var/depth_var_1_ud_samsung_mono_ground_truth_000000.mp4

video/flea3_7/sfm/train/scene7_03_flea3_7_ground_truth_000002.mp4
video/left_camera/sfm/train/scene7_03_left_camera_ground_truth_000002.mp4
video/right_camera/sfm/train/scene7_03_right_camera_ground_truth_000002.mp4
video/samsung_mono/sfm/train/scene7_03_samsung_mono_ground_truth_000002.mp4

video/flea3_7/sfm/train/seq_1_5_flea3_7_ground_truth_000001.mp4
video/left_camera/sfm/train/seq_1_5_left_camera_ground_truth_000001.mp4
video/right_camera/sfm/train/seq_1_5_right_camera_ground_truth_000001.mp4
```

## Generation Tools
All generation tools are located in `evimo/tools/evimo2_generation`.

### Clear
Deletes all generated files but leaves the raw recordings

Clear all:
```bash
./clear_all.sh ~/EVIMO/raw
```

Clear a specific recording:
```bash
./clear.sh ~/EVIMO/raw/imo/eva/scene13_dyn_test_00
```

### Generate
Runs for each camera in a sequence folder
* the offline tool to generate txt format
* the evimo-gen python tool to generat npz format and visualization frames
* ffmpeg to make the visualization video
* cleans up all intermediate artifacts to save TB's of disk space as it goes along
* all final artifacts are left in each sequences folder, they will be moved into the final dataset file/folder structure later

Generate all:
```bash
./generate_all.sh ~/raw
```

Generate a specific recording:
```
./generate.sh ~/EVIMO/raw/imo/eva/scene13_dyn_test_00
```

### Package

Checks that files that should have been made by `generate.sh` are present and copies or moves files into the released file/folder structure.

To do a dry run (check for missing generated files)

```bash
./package_all.py ~/EVIMO/raw ~/EVIMO/packaged dry
```

To do a real run (moves files and copies those that can't be moved):

Moving instead of copying saves over a 1 TB of drive space and makes the process fit on a 4TB drive.

```bash
./package_all.py ~/EVIMO/raw ~/EVIMO/packaged move
```

### Compress
Checks that files that should have been made by `generate.sh` are present and copies or moves files into the released file/folder structure.

To do a dry run:

```bash
./compress_packaged.py ~/EVIMO/packaged ~/EVIMO/compressed dry
```

To do a real run (moves files and copies those that can't be moved):

```bash
./compress_packaged.py ~/EVIMO/packaged ~/EVIMO/compressed compress
```
