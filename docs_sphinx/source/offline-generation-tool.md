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

The parameters of the `.launch` file are:
 - `folder:=<string>` - path to the [dataset configuration folder](https://github.com/better-flow/evimo/wiki/Dataset-Configuration-Folder)
 - `camera_name:=<string>` - name of the folder within the dataset configuration folder which contains camera configuration
 - `show:=<integer>` - `-1` disables the visualization; '-2' allows to inspect the sequence frame-by-frame'; positive numbers simultaneously open multiple windows at different time instances and a trajectory plot (`show:=10` will render data at 10 separate timestamps).
 - `generate:=<bool>` - if `true`, the ground truth will be saved in the camera folder (in `ground_truth` subfolder) within the dataset configuration folder.
 - `save_3d:=<bool>` - experimental feature; will save events as a `.ply` file (after filtering).
 - `fps:=<float>` - the tool will attempt to generate ground truth every 1/fps seconds; the actual frame rate or timestamps are not guaranteed - ground truth is only generated when both events (or camera frame) and Vicon pose is available; the actual timestamp will 'stick' to the lowest rate sensor (either camera or Vicon).
 - `t_offset:=<float>` - skip that many seconds from the beginning of the recording before any processing
 - `t_len:=<float>` - process at most that many seconds (negative value will process all the recording)

After the generation, the ground truth can be found in: <br>
`<dataset configuration folder>/<camera_name>/ground_truth`

The folder contains the TXT version of the sequence.
