# Raw Sequence Folder Structure

Each recorded `.bag` file is placed within its own folder. This folder contains intrinsic and extrinsic parameters for each camera recorded in the `.bag` file, as well as time offsets and a list of recorded objects on the scene.

The example of the dataset configuration folder (without a .bag file) can be found [here](https://github.com/better-flow/evimo/tree/master/evimo/config). The configuration folder should contain:
 - `objects.txt` - a file listing the recorded objects
 - one or several *camera configuration folders*
 - `<folder_name>.bag` - a bag file with the same name as the sequence's folder

An example of the file tree for a raw sequence's folder is below:

```
scene13_dyn_test_00
├── flea3_7
│   ├── calib.txt
│   ├── extrinsics.txt
│   └── params.txt
├── left_camera
│   ├── calib.txt
│   ├── extrinsics.txt
│   └── params.txt
├── objects.txt
├── right_camera
│   ├── calib.txt
│   ├── extrinsics.txt
│   └── params.txt
├── samsung_mono
│   ├── calib.txt
│   ├── extrinsics.txt
│   └── params.txt
└── scene13_dyn_test_00.bag
```

|File |Description |
--- | --- |
|`'calib.txt'`| a single line with camera intrinsic parameters in a format `fx fy cx cy k1 k2` and `k3 k4` for *equidistant* distortion model and `p1 p2` for *radtan* distortion model. See relevant code [here](https://github.com/better-flow/evimo/blob/02db52855d11907ccd8494b84ef8b753209f98ef/evimo/dataset.h#L441) and [here](https://github.com/better-flow/evimo/blob/02db52855d11907ccd8494b84ef8b753209f98ef/evimo/dataset.h#L657). |
|`'extrinsics.txt'`| 4 lines; first line: 6 parameters `tx ty tz roll pitch yaw` - Euler angles for camera-to-Vicon transformation. Third line: a single number, camera-to-Vicon timestamp offset in seconds. See relevant code [here](https://github.com/better-flow/evimo/blob/02db52855d11907ccd8494b84ef8b753209f98ef/evimo/dataset.h#L499).|
|`'params.txt'`| Camera's resolution, distortion model (`radtan` or `equidistant`), and associated topics|
|`'objects.txt'`| Object's in the [objects directory](https://github.com/better-flow/evimo/tree/master/evimo/objects) that should be in a sequence. [(example)](https://github.com/better-flow/evimo/blob/master/evimo/config/objects.txt) The objects that are not listed (or commented out) will not be in the ground truth, even if recorded. If the object is included in `objects.txt` but has no ROS messages, most tools will terminate with an error. To create new objects see [adding a new object](adding-a-new-object.md)|
|`'.bag'`| ROS bagfile containing the raw data for the sequence|

The following topics can be specified in `params.txt`:
|Topic |Description |
|--- | --- |
|`ros_image_topic`| ROS topic with classical rgb camera frames; can be `none`|
|`ros_event_topic`| ROS topic with events from DVS cameras; can be `none`|
|`ros_pos_topic`| ROS topic with [Vicon messages](https://github.com/KumarRobotics/vicon/tree/a6143808872ab02e8ebdc9384d4ea4d475e815b8/vicon/msg) for the camera and objects|

## Defaults
A folder of default camera configurations are provided [here](https://github.com/better-flow/evimo/tree/master/evimo/config). The defaults are used by the online tool.
