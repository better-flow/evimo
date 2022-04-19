# Calibration Tools
The calibration tools are designed to calibrate (intrinsically and extrinsically) a *single* camera and Vicon. We use two tools to perform calibration: the *collect* tool, which allows to collect a small amount of frames for a complete (but not always the most accurate) calibration, and *refine* tool for calibration refinement, given the initial calibration is provided. We run the *refine* tool before every data collection, while the *collect* tool has only been used once.

## refine tool
The code for the tool is located [here](https://github.com/better-flow/evimo/blob/9acbcc71aeb84bc720d689247dd3f075ac59b465/evimo/calib/refine_calibration.cpp#L1134). A typical use-case:
```
rosrun evimo refine _conf:=./evimo2/calib_112420/samsung_mono.cfg
```

### Input to the tool:
An input to the tool a folder with one or several `.bag` files, which contain the Vicon calibration wand recording: both camera images (or events) and Vicon pose topics for wand *and* for the sensor rig. The folder (with multiple sensor configuration files) may look like:

<img src="https://github.com/better-flow/evimo/blob/master/docs/wiki_img/refine_input_folder.png" width="600" />

The configuration file (for example `samsung_mono.cfg`) will contain some sensor-specific settings and configuration for the refinement run:
```
camera_name: samsung_mono
image_th: 240 # only useful for classical cameras
dilate_blobs: false # only useful for event cameras
bag_file: wand_01_depth_00.bag 3 20
bag_file: wand_01_depth_01.bag 2 20
bag_file: wand_01_depth_02.bag 0 -1
```
- `camera_name` - is required to extract initial calibration estimates from the [dataset configuration folder](https://github.com/better-flow/evimo/wiki/Dataset-Configuration-Folder), as well as the camera ROS topic.
- `bag_file` - list of `.bag` files to be used; one per line. The first parameter is relative path to the `.bag` file, then start offset (how many seconds to skip from the beginning of the recording), then length of the recording to process (`-1` means process all recording). In the example above, the `wand_01_depth_00.bag` is processed from sec. *3* to sec. *23*.
- `image_th` - optional parameter which is only used for classical cameras ([code](https://github.com/better-flow/evimo/blob/9acbcc71aeb84bc720d689247dd3f075ac59b465/evimo/calib/refine_calibration.cpp#L1052)). Will threshold an image at a specified value before blob extraction.
- `dilate_blobs` - optional parameter which is only used for event cameras ([code](https://github.com/better-flow/evimo/blob/9acbcc71aeb84bc720d689247dd3f075ac59b465/evimo/calib/refine_calibration.cpp#L1026)). Dilates an image with a 5x5 kernel.


In addition, the tool accepts several parameters (we recommend sticking with the defaults):
- `_conf:=<string>` - the path to the camera configuration file (such as `samsung_mono.cfg`). The name of the file is not used by the tool and can be any valid file name.
- `_folder:=<string>` - the path to the [dataset configuration folder](https://github.com/better-flow/evimo/wiki/Dataset-Configuration-Folder). By default, the `evimo/evimo/config/` is used. This provides an initial guess for intrinsics and extrinsics of the camera, as well as camera ROS topic.
- `_wand_topic:=<string>` - a ROS topic of Vicon calibration wand; by default `/vicon/Wand`.
- `_e_slice_width:=<float>` - for event cameras, width of event slice (in seconds) to use to generate an image and later detect the Wand. Large values can cause motion blur if Wand was moved fast, small values will cause detection to be less reliable. By default `0.02`.
- `_e_fps:=<float>` - frequency at which to generate 'frames' for event cameras. by default `2.0 / e_slice_width`.
- `_tracker_max_pps:=<float>` - maximum allowed 'pixel-per-second' speed of a matched blob on adjacent frames; essentially a threshold on distance between blobs on adjacent frames (normalized by framerate) to be considered the same blob. By default `1000 * std::max(res_y, res_x) / 640.0`.
- `_tracker_min_len:=<float>` - the smallest length of track for a tracked blob to be considered a 'reliable' track. Default is `0.3`.

**Note**: for event cameras the tool performs *frequency filtering* ([code](https://github.com/better-flow/evimo/blob/9acbcc71aeb84bc720d689247dd3f075ac59b465/evimo/calib/refine_calibration.cpp#L1009)). We recorded the calibration for *evimo* with Vicon wand leds flickering at 200Hz. You may wish to disable this feature in case your recording does not use flickering leds.

### Refinement pipeline:
The tool extracts the tracks (wand led markers) separately from each `.bag` file, but then uses them all together in the optimization. **Note**: vicon tracks IR markers on the wand, while the detected markers are in visible light, and are offset from IR markers. We used a 3D scan of the wand ([link](https://github.com/better-flow/evimo/blob/9acbcc71aeb84bc720d689247dd3f075ac59b465/evimo/objects/wand/model.ply)) to extract the offset. The mapping can be found [here](https://github.com/better-flow/evimo/blob/9acbcc71aeb84bc720d689247dd3f075ac59b465/evimo/calib/detect_wand.h#L237).

 1) The data is preprocessed: events are frequency-filtered, and event slices are downprojected to form images. OpenCV's blob extractor is then used to extract blobs for every individual frame ([code](https://github.com/better-flow/evimo/blob/9acbcc71aeb84bc720d689247dd3f075ac59b465/evimo/calib/refine_calibration.cpp#L1001)). 
 2) The Vicon tracks are converted to rig frame (also tracked by Vicon), and visible (red) led locations are computed from IR led poses ([code](https://github.com/better-flow/evimo/blob/9acbcc71aeb84bc720d689247dd3f075ac59b465/evimo/calib/refine_calibration.cpp#L941)).
 3) The blobs are tracked (nearest neighbor match with thresholding) and short tracks are removed ([code](https://github.com/better-flow/evimo/blob/9acbcc71aeb84bc720d689247dd3f075ac59b465/evimo/calib/refine_calibration.cpp#L60), [code](https://github.com/better-flow/evimo/blob/9acbcc71aeb84bc720d689247dd3f075ac59b465/evimo/calib/refine_calibration.cpp#L1073)).
 4) The wand is detected [here](https://github.com/better-flow/evimo/blob/9acbcc71aeb84bc720d689247dd3f075ac59b465/evimo/calib/detect_wand.h#L249), and the marker labels are propagated along tracks.
 5) Cross-correlation between Vicon reprojected markers (using initial calibration estimate) and detected markers is used to align time per each bag file ([code](https://github.com/better-flow/evimo/blob/9acbcc71aeb84bc720d689247dd3f075ac59b465/evimo/calib/refine_calibration.cpp#L532)).
 6) The calibration is [ran twice](https://github.com/better-flow/evimo/blob/9acbcc71aeb84bc720d689247dd3f075ac59b465/evimo/calib/refine_calibration.cpp#L1232) after the initial calibration the outlier points are removed (the points with error below mean), and the calibration is repeated.

### Refinement result:
The tool will not save the result of the calibration, but will output it to the terminal. It can be directly copied to the `camera folder` within the [dataset configuration folder](https://github.com/better-flow/evimo/wiki/Dataset-Configuration-Folder).

In addition, the tool will plot the statistics on the data:

<img src="https://github.com/better-flow/evimo/blob/master/docs/wiki_img/refine_tool.png" width="1000" />

Top row: input (estimate *after* initial calibration / before outlier removal); left: all x points, right: all y points. Middle row: same as top, but after the final refinement step (if you would like to generate a similar plot *before* the calibration, change `false` to `true` [here](https://github.com/better-flow/evimo/blob/9acbcc71aeb84bc720d689247dd3f075ac59b465/evimo/calib/refine_calibration.cpp#L1232)). Bottom left: distribution of depth across all recordings. Bottom right: distribution of x-y points projected on the camera plane.


## collect tool
```
roslaunch evimo collect.launch config:=~/config.txt output_dir:=/tmp/collect
```

<img src="https://github.com/better-flow/evimo/blob/master/docs/wiki_img/collect_tool.png" width="1000" />


```
rig_0:   vicon   /vicon/dvs_rig
wand:    vicon   /vicon/Wand

cam_0:    rgb     /flea3/image_color/compressed
cam_1:    event   /prophesee/left/cd_events_buffer
cam_2:    event   /prophesee/right/cd_events_buffer
cam_3:    event   /samsung/camera/events
```

<img src="https://github.com/better-flow/evimo/blob/master/docs/wiki_img/collect_out.png" width="1000" />



```
evimo/evimo/calib$ ./wand_calib.py /home/ncos/ooo/EVIMO2/recording_day_0/calib/samsung_00/ /home/ncos/ooo/EVIMO2/recording_day_0/calib/samsung_01/ /home/ncos/ooo/EVIMO2/recording_day_0/calib/samsung_02/ /home/ncos/ooo/EVIMO2/recording_day_0/calib/samsung_03/ /home/ncos/ooo/EVIMO2/recording_day_0/calib/samsung_04/ -c cam_3
```

<img src="https://github.com/better-flow/evimo/blob/master/docs/wiki_img/collect_result.png" width="1000" />

