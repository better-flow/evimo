# Dataset Configuration Folder

Each recorded `.bag` file is placed within its own 'dataset configuration folder'. This folder contains intrinsic and extrinsic parameters for each camera recorded in the `.bag` file, as well as camera-to-motion capture time offsets and a list of recorded objects on the scene.

The example of the dataset configuration folder (without a .bag file) can be found [here](https://github.com/better-flow/evimo/tree/master/evimo/config). The configuration folder should contain:
 - `objects.txt` - a file listing the recorded objects.
 - one or several *camera configuration folders*.
 - `<folder_name>.bag` - a bag file with the same name as the name of the dataset configuration folder.

A typical dataset folder may look like this:

<img src="https://github.com/better-flow/evimo/blob/master/docs/wiki_img/dataset_folder.png" width="500" />


## objects.txt ([example](https://github.com/better-flow/evimo/blob/master/evimo/config/objects.txt))
This file lists object folders from [objects](https://github.com/better-flow/evimo/tree/master/evimo/objects) directory (see [adding a new object](https://github.com/better-flow/evimo/wiki/Adding-a-New-Object)), one object per line; example: `objects/toy_00`. The objects that are not listed (or commented out) will not be in the ground truth, even if recorded. If the object is included in `objects.txt` but has no ROS messages, most tools will terminate with an error.


## camera configuration folder ([example](https://github.com/better-flow/evimo/tree/master/evimo/config/samsung_mono))

<img src="https://github.com/better-flow/evimo/blob/master/docs/wiki_img/camera_configuration.png" width="500" />


The camera configuration folder contains 3 configuration files:
 - `calib.txt`: a single line with camera intrinsic parameters in a format `fx fy cx cy k1 k2` and `k3 k4` for *equidistant* distortion model and `p1 p2` for *radtan* distortion model. See relevant code [here](https://github.com/better-flow/evimo/blob/02db52855d11907ccd8494b84ef8b753209f98ef/evimo/dataset.h#L441) and [here](https://github.com/better-flow/evimo/blob/02db52855d11907ccd8494b84ef8b753209f98ef/evimo/dataset.h#L657).
 - `extrinsics.txt`: 4 lines; first line: 6 parameters `tx ty tz roll pitch yaw` - Euler angles for camera-to-Vicon transformation. Third line: a single number, camera-to-Vicon timestamp offset in seconds. See relevant code [here](https://github.com/better-flow/evimo/blob/02db52855d11907ccd8494b84ef8b753209f98ef/evimo/dataset.h#L499).
 - `params.txt`: 6 lines in a `key: value` format (note, in OpenCV resolution is specified using an opposite convention)
   - `res_x` : camera resolution along x axis (typically smaller)
   - `res_y` : camera resolution along y axis (typically larger)
   - `dist_model`: either `radtan` or `equidistant`
   - `ros_image_topic`: ROS topic where classical rgb camera frames can be expected; can be `none`
   - `ros_event_topic`: ROS topic where events from event-based cameras can be expected; can be `none`
   - `ros_pos_topic`: ROS topic with [Vicon messages](https://github.com/KumarRobotics/vicon/tree/a6143808872ab02e8ebdc9384d4ea4d475e815b8/vicon/msg) for the camera (or sensor rig)

