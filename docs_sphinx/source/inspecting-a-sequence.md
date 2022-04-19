# Inspecting a Sequence

This page describes how to use the **offline tool** for sequence visualization. The input to the tool is a [dataset configuration folder](https://github.com/better-flow/evimo/wiki/Dataset-Configuration-Folder) with a recorded `.bag` file, which contains ROS topics with camera frames or events and [vicon messages](https://github.com/KumarRobotics/vicon/tree/a6143808872ab02e8ebdc9384d4ea4d475e815b8/vicon/msg).

The tutorial assumes that [ROS](http://wiki.ros.org/noetic/Installation/Ubuntu) has been installed, and the *evimo* repository cloned into [catkin workspace](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment).

# Offline Tool Parameters
The parameters for the tool are read [here](https://github.com/better-flow/evimo/blob/67bc579e0588f8acd9d2cb37eb85cd5ddac8b6ef/evimo/offline.cpp#L173). When running with `rosrun`, parameters are:
 - `_folder:=<string>` - path to the [dataset configuration folder](https://github.com/better-flow/evimo/wiki/Dataset-Configuration-Folder)
 - `_camera_name:=<string>` - name of the folder within the dataset configuration folder which contains camera configuration
 - `_show:=<integer>` - `-1` disables the visualization; '-2' allows to inspect the sequence frame-by-frame'; positive numbers simultaneously open multiple windows at different time instances and a trajectory plot (`show:=10` will render data at 10 separate timestamps).
 - `_generate:=<bool>` - if `true`, the ground truth will be saved in the camera folder (in `ground_truth` subfolder) within the dataset configuration folder.
 - `_save_3d:=<bool>` - experimental feature; will save events as a `.ply` file (after filtering). Code [here](https://github.com/better-flow/evimo/blob/67bc579e0588f8acd9d2cb37eb85cd5ddac8b6ef/evimo/annotation_backprojector.h#L156).
 - `_fps:=<float>` - the tool will attempt to generate ground truth every 1/fps seconds; the actual frame rate or timestamps are not guaranteed - ground truth is only generated when both events (or camera frame) and Vicon pose is available; the actual timestamp will 'stick' to the lowest rate sensor (either camera or Vicon).
 - `_start_time_offset:=<float>` - skip that many seconds from the beginning of the recording before any processing
 - `_sequence_duration:=<float>` - process at most that many seconds (negative value will process all the recording)

## Visualization modes
Two kinds of visualization are currently supported: 
1) `_show:=-2`: a frame-by-frame visualization (with a scroll bar to go through the dataset frames); rendering modes can be changed by pressing keys 1-4; from left to right: mask with events overlaid, mask, depth with events overlaid, events color-coded by their timestamp:
<img src="https://github.com/better-flow/evimo/blob/master/docs/wiki_img/mode_1234.png" width="1000" />

The visualization code is [here](https://github.com/better-flow/evimo/blob/67bc579e0588f8acd9d2cb37eb85cd5ddac8b6ef/evimo/offline.cpp#L126).

2) `_show:=<any positive integer>` a multi-frame visualization, this does not allow to scroll through the sequence but allows to see multiple frames at once. Additionally, the trajectory plots are shown (red vertical lines are at the timestamps of rendered frames) and the manual calibration window, which allows to adjust the extrinsics manually. **Note** the rotation component in trajectories is plotted in Euler angles, and angles can 'roll over' from *-pi* to *pi* during the full turn.
<img src="https://github.com/better-flow/evimo/blob/master/docs/wiki_img/mode_5.png" width="1000" />

The visualization code is [here](https://github.com/better-flow/evimo/blob/67bc579e0588f8acd9d2cb37eb85cd5ddac8b6ef/evimo/dataset_frame.h#L82).

## Keyboard Shortcuts
General ([code](https://github.com/better-flow/evimo/blob/67bc579e0588f8acd9d2cb37eb85cd5ddac8b6ef/evimo/dataset.h#L196)):
 - `'esc'` - exit the application
 - `'1'` - display mask + events or mask + rgb image
 - `'2'` - display mask
 - `'3'` - display depth + events (there is no rgb overlay)
 - `'4'` - display events color coded by the timestamp within slice
 - `'space'` - next rendering mode (1/2/3/4)
 - `'['` - decrease event slice width
 - `']'` - increase event slice width
 - `'o'` - (experimental) increase window for Vicon pose smoothing (default 0)
 - `'p'` - (experimental) decrease window for Vicon pose smoothing
 - `'c'` - reset extrinsic calibration sliders to mid value (sliders only available in multi-frame mode)
 - `'s'` - write the calibration (extrinsic or time offset) to the [camera folder](https://github.com/better-flow/evimo/wiki/Dataset-Configuration-Folder)

In frame-by-frame mode only ([code](https://github.com/better-flow/evimo/blob/67bc579e0588f8acd9d2cb37eb85cd5ddac8b6ef/evimo/offline.cpp#L107)):
 - `'''` - move one frame forward
 - `';'` - move one frame backward
 - `'c'` - render a 3d pointcloud (experimental - [additional controls](https://github.com/better-flow/evimo/blob/67bc579e0588f8acd9d2cb37eb85cd5ddac8b6ef/evimo/annotation_backprojector.h#L661))

