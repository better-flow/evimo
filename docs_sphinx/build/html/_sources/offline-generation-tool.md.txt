# Offline Generation Tool
This page describes how to use the **offline tool** - the main tool to generate ground truth from a recording. The input to the tool is a [dataset configuration folder](https://github.com/better-flow/evimo/wiki/Dataset-Configuration-Folder) with a recorded `.bag` file, which contains ROS topics with camera frames or events and [vicon messages](https://github.com/KumarRobotics/vicon/tree/a6143808872ab02e8ebdc9384d4ea4d475e815b8/vicon/msg).

A [ROS](http://wiki.ros.org/noetic/Installation/Ubuntu) environment and workspace that is analagous to the provided [Docker environment](docker-environment.md) is required.

The tool is only capable of running per-camera; a typical `roslaunch` command may look like:
```
    roslaunch evimo event_imo_offline.launch show:=-1 folder:=<dataset configuration folder> camera_name:=<camera_name> \
                                             generate:=true \
                                             save_3d:=false \
                                             fps:=40 \
                                             t_offset:=0 \
                                             t_len:=-1
```

After the generation, the TXT version of the ground truth can be found in: <br>
`<dataset configuration folder>/<camera_name>/ground_truth`

## Parameters
The parameters of the `.launch` file are:

|Key |Description |
------------ | --- |
|`folder:=<string>`| Path to the [dataset configuration folder](raw-sequence-structure.md)|
|`camera_name:=<string>`| Name of the folder within the dataset configuration folder<br> that contains the camera configuration|
|`show:=<integer>`| `-1` disables the visualization <br>`-2` allows to inspect the sequence frame-by-frame'<br>Positive numbers show the trajectory and a collection of evenly <br> spaced frames (e.g. `show:=10` will render data at 10 timestamps). <br> A full description of the visualization modes is available [here](raw-sequence-inspection.md)|
|`generate:=<bool>`| If `true`, the ground truth will be saved in the camera folder <br> (in `ground_truth` subfolder) within the dataset configuration folder.|
|`save_3d:=<bool>`| Experimental feature that will save filtered events as a `.ply` file|
|`fps:=<float>`| The tool will attempt to generate ground truth every 1/fps seconds <br><br> In EVIMO2v2 the frame rate and timestamps are gaurunteed for<br> event cameras. For classical cameras, the frametimes determine<br> the ground truth times. <br><br> In EVIMO and EVIMO2v1 the actual frame rate or timestamps<br> are not guaranteed. Ground truth is only generated when both<br> events (or camera frame) and Vicon pose are available. E.g. the<br> actual timestamp will 'stick' to the lowest rate data source.|
|`t_offset:=<float>`|Skip `t_offset` seconds from the beginning of the recording|
|`t_len:=<float>`| Process at most `t_len` seconds<br>Negative values cause the entire recording to be processed|
