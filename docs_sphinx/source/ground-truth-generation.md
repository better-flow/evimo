# Ground Truth Generation
This page describes how to use the **offline tool** - the main tool to generate ground truth from a recording. The input to the tool is a [dataset configuration folder](https://github.com/better-flow/evimo/wiki/Dataset-Configuration-Folder) with a recorded `.bag` file, which contains ROS topics with camera frames or events and [vicon messages](https://github.com/KumarRobotics/vicon/tree/a6143808872ab02e8ebdc9384d4ea4d475e815b8/vicon/msg).

The tutorial assumes that [ROS](http://wiki.ros.org/noetic/Installation/Ubuntu) has been installed, and the *evimo* repository cloned into [catkin workspace](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment).

## Generating Ground Truth
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
 - `t_len`:=<float>` - process at most that many seconds (negative value will process all the recording)

## Ground Truth Format
After the generation, the ground truth can be found in `<dataset configuration folder>/<camera_name>/ground_truth`. The folder (initially) will contain:
 - `depth_mask_<frame_id>.png` - the `.png` 16-bit images with depth and mask channels. Depth is in `mm.` (integer); mask consists of per-pixel object ids multiplied by *1000*. To see/change object ids see [adding a new object](https://github.com/better-flow/evimo/wiki/Adding-a-New-Object).
   - A Python example on how to read the image can be found [here](https://github.com/better-flow/pydvs/blob/bdff8de0c3c7df24d3143154b415091d88a1e4c2/samples/evimo-gen.py#L229) and [here](https://github.com/better-flow/pydvs/blob/bdff8de0c3c7df24d3143154b415091d88a1e4c2/samples/evimo-gen.py#L11).
   - The relevant C++ code is [here](https://github.com/better-flow/evimo/blob/3b20a8a3ee729b9b9eb69bda05ac4cbf8b9773cb/evimo/dataset_frame.h#L300).
 - `img_<frame_id>.png` - when available, frames from the classical camera.
 - `meta.txt` - a `.json`-like file which contains camera and object poses, calibration parameters as well as sensor timestamps for each object id. 
   - A special object id `'cam'` is used for camera. Note that `'vel'` key is experimental; we recommend that the velocity is computed separately. The `'gt_frame'` key specifies the name of the relevant 'depth_mask_<>.png' file. Optionally, `'classical_frame'` key specifies the name of the relevant classical rgb frame (when available).
   - **Note:** the `meta.txt` contains two duplicates of the Vicon trajectory: one with ground truth (the key is `'frames'`) - generated at the frame rate of ground truth. Another - `'full_trajectory'` has Vicon messages at full rate (200Hz), but keys `'id'` and `'gt_frame'` cannot be used.
   - `meta.txt` can be easily read by Python `eval()` method. See example [here](https://github.com/better-flow/pydvs/blob/bdff8de0c3c7df24d3143154b415091d88a1e4c2/samples/evimo-gen.py#L182).
   - An example of the contents of the `meta.txt`:
   ```
   {
   'meta': {'fx': 519.299927, 'fy': 515.430115, 'cx': 328.382660, 'cy': 245.452011, 'k1': 0.091035, 'k2': -0.118269, 'k3': 0.000000, 'k4': 0.000000, 'p1': 0.004237, 'p2': 0.004445, 'res_x': 640, 'res_y': 480, 'dist_model': 'radtan'}
   , 'frames': [
   {
   'id': 18446744073709551615,		'ts': 5.446962,
   'cam': {
	'vel': {'t': {'x': -0.550138, 'y': 0.452351, 'z': 0.056179}, 'rpy': {'r': -0.040196, 'p': -0.719177, 'y': -0.282081}, 'q': {'w': 0.925566, 'x': -0.068081, 'y': -0.345680, 'z': -0.138557}},
	'pos': {'t': {'x': 0.463163, 'y': -0.075241, 'z': 0.107412}, 'rpy': {'r': -0.075974, 'p': 0.150307, 'y': 0.201305}, 'q': {'w': 0.991128, 'x': -0.045218, 'y': 0.070844, 'z': 0.102964}},
	'ts': 5.446962},
   '9': {
	'pos': {'t': {'x': -0.493914, 'y': 0.004656, 'z': 0.491391}, 'rpy': {'r': -0.010523, 'p': -0.556265, 'y': -0.785069}, 'q': {'w': 0.887870, 'x': -0.109701, 'y': -0.251739, 'z': -0.369160}},
	'ts': 5.446962},
   '22': {
	'pos': {'t': {'x': -0.389233, 'y': -0.082026, 'z': 0.763271}, 'rpy': {'r': 1.090704, 'p': 0.085538, 'y': 0.037217}, 'q': {'w': 0.854428, 'x': 0.517475, 'y': 0.046191, 'z': -0.006281}},
	'ts': 5.446962},
   'gt_frame': 'depth_mask_18446744073709551615.png'
   },
   ```
 - `events.txt` - a text file with events from an event camera (when available), a single event per line in a format `<timestamp px py polarity>`; The contents may look like:
   ```
   0.001776896 624 3 0
   0.001776896 624 87 0
   0.001776896 624 141 0
   0.001776896 624 151 0
   0.001776896 624 153 0
   0.001776896 624 460 1
   0.001776896 625 154 0
   0.001776896 625 155 0
   0.001776896 625 458 1
   0.001776896 626 146 0
   ```

