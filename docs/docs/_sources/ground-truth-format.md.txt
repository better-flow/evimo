# Ground Truth Format

The ground truth is available in the following formats:

* `NPZ` - A collection of [`.npy` files](https://numpy.org/doc/stable/reference/generated/numpy.save.html), supported by [NumPy](https://numpy.org/doc/stable/index.html)
    * EVIMO2v2 - Each sequence is a collection of `.npz` and `.npy` files
    * EVIMO, EVIMO2v1 - Each sequence is compressed into a single `.npz` file
* `TXT` - Ground truth and recordings in the form of `.png` and `.txt` files

The high level differences between the formats are described [here](#format-comparison).

## NPZ (EVIMO2v2)
There is one folder per sequence. A sequences folder can be found in the following path:<br>
`<camera>/<category>/<subcategory>/<sequence name>`

Inside a sequences folder are the following files:

|File |Description |
--- | --- |
|`dataset_classical.npz`| Dictionary of *(RES_Y, RES_X)* arrays with keys `classical_<frame id>`|
|`dataset_depth.npz`| Dictionary of *(RES_Y, RES_X)* arrays with keys `depth_<frame id>`|
|`dataset_mask.npz`| Dictionary of *(RES_Y, RES_X)* arrays with keys `mask_<frame id>`<br>masks contains object ids multiplied by `1000`|
|`dataset_events_p.npy`| Array of *(NUM_EVENTS, 1) containing events polarity <br> **Can be memory mapped** <br> **Samsung event polarity is inverted compared to Prophesee**|
|`dataset_events_t.npy`| Array of *(NUM_EVENTS, 1) containing events time<br> **Can be memory mapped**|
|`dataset_events_xy.npy`| Array of *(NUM_EVENTS, 2) containg events pixel location<br> **Can be memory mapped**|
|`dataset_info.npz`| Dictionary of arrays `D`, `discretization`, `index`, `K`, `meta`<br> Contents are identical to the EVIMO, EVIMO2v1 NPZ format|

Ground truth depth and masks are not evenly spaced in time because the Vicon system sometimes loses track due to occlusion. The `meta` field requires all the required timestamping information to handle the irregular sampling.

In EVIMO2v2 the classical camera will have a different number of depth/mask frames and classical frames because classical frames are kept even if the depth and mask are unavailable.

## NPZ (EVIMO and EVIMO2v1)
There is one compressed `.npz` file per sequence. A sequences file can be found in the following path: `<camera>/<category>/<subcategory>/<sequence name>.npz`

A sequences `.npz` file contains the following `.npy` files:

|File |Description |
--- | --- |
|`classical.npy`| Array with shape *(NUM_FRAMES, RES_Y, RES_X)*<br> Contains classical frames if available|
|`depth.npy`| Array of shape *(NUM_FRAMES, RES_Y, RES_X)*<br>depth is in *mm*|
|`mask.npy` | Array of shape *(NUM_FRAMES, RES_Y, RES_X)*<br>masks contains object ids multiplied by `1000`|
|`events.npy`| Array of shape *(NUM_EVENTS, 4)*<br>Each row contains an events timestamp, x/y pixel coordinate and polarity <br> **EVIMO2: Samsung event polarity is inverted compared to Prophesee**
|`meta.npy`| Python dictionary containing intrinsics, timestamps, poses, and IMU data<br>The full description is [here](ground-truth-format.md#metas-contents)
|`K.npy/D.npy`| Intrinsic and distortion parameters, also available in `meta.npy`|
|`index.npy`| A helper lookup table for fast timestamp-to-event index computation<br>Contains indices of events every `discretization.npy` seconds
|`discretization.npy`| The time between events corresponding to the indices in `index.npy`|

Ground truth depth and masks are not evenly spaced in time because the Vicon system sometimes loses track due to occlusion. The `meta` field requires all the required timestamping information to handle the irregular sampling.

In EVIMO and EVIMO2v1 the classical camera frames are only available when depth and mask frames are available.

## TXT
There is one folder per sequence. A sequences folder can be found in the following path:<br>
`<camera>/<category>/<subcategory>/<sequence name>`

We strongly suggest using the [pydvs sample](https://github.com/better-flow/pydvs/blob/bdff8de0c3c7df24d3143154b415091d88a1e4c2/samples/evimo-gen.py) as a prototype for manipulating the TXT data. This is the script which converts the TXT format (the output of the [C++ pipeline](offline-generation-tool.md) to NPZ. It also generates the trajectory plot distributed with the TXT format, event slices, and frames for visualization videos.

|Item |Description |
--- | --- |
|`img/img_<frame id>.png`| Conventional frames from a classical camera, when available|
|`img/depth_mask_<frame id>.png`| 16-bit png's with depth and masks in different channels <br>Depth is in *mm*<br> Masks contain per-pixel object ids multiplied by `1000`<br>An example of reading the image is available [here](https://github.com/better-flow/pydvs/blob/bdff8de0c3c7df24d3143154b415091d88a1e4c2/samples/evimo-gen.py#L229)<br>The relevant C++ code is [here](https://github.com/better-flow/evimo/blob/3b20a8a3ee729b9b9eb69bda05ac4cbf8b9773cb/evimo/dataset_frame.h#L300)|
|`events.txt`| File with events (if available)<br>One event per line in a format `<timestamp px py polarity>`|
|`calib.txt`<br>`extrinsics.txt`<br> `params.txt` or `config.txt`| Camera parameters see [here](raw-sequence-structure.md) for details|
|`meta.txt`| String containing intrinsics, timestamps, poses, and IMU data<br>The full description is [here](ground-truth-format.md#metas-contents)|
|`position_plots.pdf`| A plot of camera/object trajectories for visualization only|

Ground truth depth and masks are not evenly spaced in time because the Vicon system sometimes loses track due to occlusion. The `meta` field requires all the required timestamping information to handle the irregular sampling.

In EVIMO and EVIMO2v1 the classical camera frames are only available when depth and mask frames are available.

In EVIMO2v2 the classical camera will have a different number of depth/mask frames and classical frames because classical frames are kept even if the depth and mask are unavailable.


## Meta's Contents

|Key |Description |
--- | --- |
|`'frames'`| Array of dictionaries with one entry per ground truth sampling period <br>Each dictionary contains the pose of each object and the camera<br> See [here](ground-truth-format.md#transform-convention) for the transform conventions <br> A special object id `'cam'` is used for the camera <br> `'gt_frame'`/`'classical_frame'` denote the pose's ground truth/classical .png files<br>`'id'` denotes indies to the ground truth/classical frame in the NPZ format |
|`'full_trajectory'`| Array of dictionaries with one entry per Vicon pose measurement (200Hz) |
|`imu`| Dictionary of arrays of IMU samples<br> One array per IMU on the EVIMO2 camera rig <br> No IMU data is available for EVIMO|
|`'meta'` | Camera intrinsics and time offset from the corresponding ROS bag file.<br>See [here](raw-sequence-structure.md) for details. |

**Note:** The `'vel'` key is available in older versions of the dataset, it should be ignored.

An example of the contents of `meta.npy` or `meta.txt` is given below. In either case, the structure is a Python dictionary and the text version can be read with Python's `eval()` method as shown [here](https://github.com/better-flow/pydvs/blob/bdff8de0c3c7df24d3143154b415091d88a1e4c2/samples/evimo-gen.py#L182).

For brevity, only a single element of the `'frames'`, `'full_trajectory'`, and IMU arrays are shown.

```
{'frames': [{'11': {'pos': {'q': {'w': 0.27729, 'x': 0.145438, 'y': -0.40157, 'z': 0.860639},
                            'rpy': {'p': -0.49274, 'r': -0.765635, 'y': 2.720059},
                            't': {'x': -0.005627, 'y': 0.338652, 'z': 1.391626}},
                    'ts': 0.031794},
             '14': {'pos': {'q': {'w': 0.507373, 'x': 0.528164, 'y': 0.658874, 'z': 0.171756},
                            'rpy': {'p': 0.508834, 'r': 2.080551, 'y': 1.487373},
                            't': {'x': -0.204377, 'y': 0.395282, 'z': 1.558496}},
                    'ts': 0.031794},
             '15': {'pos': {'q': {'w': 0.094314, 'x': -0.133898, 'y': 0.853899, 'z': 0.493997},
                            'rpy': {'p': 0.29774, 'r': 2.114005, 'y': -2.999389},
                            't': {'x': -0.016263, 'y': 0.39068, 'z': 1.34646}},
                    'ts': 0.031794},
             '22': {'pos': {'q': {'w': 0.390027, 'x': 0.751193, 'y': -0.092049, 'z': -0.524514},
                            'rpy': {'p': 0.79837, 'r': 1.780868, 'y': -0.901796},
                            't': {'x': -0.039562, 'y': 0.17172, 'z': 1.283998}},
                    'ts': 0.031794},
             '23': {'pos': {'q': {'w': -0.315782, 'x': 0.541274, 'y': 0.711081, 'z': 0.318855},
                            'rpy': {'p': -0.917802, 'r': 2.956836, 'y': 1.931814},
                            't': {'x': -0.128985, 'y': -0.108972, 'z': 0.90992}},
                    'ts': 0.031794},
             '24': {'pos': {'q': {'w': -0.302069, 'x': -0.559534, 'y': 0.146147, 'z': 0.757837},
                            'rpy': {'p': 0.862972, 'r': 1.036446, 'y': -1.869526},
                            't': {'x': 0.232299, 'y': -0.171992, 'z': 0.765111}},
                    'ts': 0.031794},
             '5': {'pos': {'q': {'w': 0.897532, 'x': -0.232753, 'y': -0.370827, 'z': -0.052436},
                           'rpy': {'p': -0.761582, 'r': -0.551012, 'y': 0.108671},
                           't': {'x': 0.24287, 'y': -0.042829, 'z': 0.686142}},
                   'ts': 0.031794},
             '6': {'pos': {'q': {'w': 0.629238, 'x': -0.435374, 'y': -0.09361, 'z': -0.636982},
                           'rpy': {'p': -0.737523, 'r': -0.61769, 'y': -1.337676},
                           't': {'x': 0.091899, 'y': -0.008143, 'z': 1.013714}},
                   'ts': 0.031794},
             'cam': {'pos': {'q': {'w': 1.0, 'x': 3.6e-05, 'y': 0.000342, 'z': 0.000158},
                             'rpy': {'p': 0.000683, 'r': 7.2e-05, 'y': 0.000316},
                             't': {'x': -0.000103, 'y': -0.000202, 'z': 2.9e-05}},
                     'ts': 0.031794},
             'classical_frame': 'img_0.png',
             'gt_frame': 'depth_mask_0.png',
             'id': 0,
             'ts': 0.031794}],
 'full_trajectory': [{'11': {'pos': {'q': {'w': 0.27728, 'x': 0.145302, 'y': -0.401452, 'z': 0.86072},
                                     'rpy': {'p': -0.492419, 'r': -0.765375, 'y': 2.719923},
                                     't': {'x': -0.005162, 'y': 0.338606, 'z': 1.391609}},
                             'ts': 0.017647},
                      '14': {'pos': {'q': {'w': 0.507253, 'x': 0.527763, 'y': 0.659383, 'z': 0.17139},
                                     'rpy': {'p': 0.509845, 'r': 2.081517, 'y': 1.488579},
                                     't': {'x': -0.203892, 'y': 0.395231, 'z': 1.558574}},
                             'ts': 0.017647},
                      '15': {'pos': {'q': {'w': 0.093911, 'x': -0.133816, 'y': 0.853955, 'z': 0.493998},
                                     'rpy': {'p': 0.296946, 'r': 2.11404, 'y': -2.999136},
                                     't': {'x': -0.015858, 'y': 0.390641, 'z': 1.346457}},
                             'ts': 0.017647},
                      '22': {'pos': {'q': {'w': 0.390079, 'x': 0.751092, 'y': -0.092001, 'z': -0.524628},
                                     'rpy': {'p': 0.798503, 'r': 1.780428, 'y': -0.90209},
                                     't': {'x': -0.039156, 'y': 0.171642, 'z': 1.284023}},
                             'ts': 0.017647},
                      '23': {'pos': {'q': {'w': -0.31792, 'x': 0.540927, 'y': 0.710404, 'z': 0.318828},
                                     'rpy': {'p': -0.921696, 'r': 2.960192, 'y': 1.930286},
                                     't': {'x': -0.128655, 'y': -0.10911, 'z': 0.90998}},
                             'ts': 0.017647},
                      '24': {'pos': {'q': {'w': -0.302236, 'x': -0.559342, 'y': 0.146205, 'z': 0.757901},
                                     'rpy': {'p': 0.862507, 'r': 1.036064, 'y': -1.869722},
                                     't': {'x': 0.232548, 'y': -0.172056, 'z': 0.765072}},
                             'ts': 0.017647},
                      '5': {'pos': {'q': {'w': 0.897617, 'x': -0.232821, 'y': -0.370598, 'z': -0.052295},
                                    'rpy': {'p': -0.761021, 'r': -0.551155, 'y': 0.108874},
                                    't': {'x': 0.243082, 'y': -0.04289, 'z': 0.68607}},
                            'ts': 0.017647},
                      '6': {'pos': {'q': {'w': 0.62919, 'x': -0.435517, 'y': -0.093582, 'z': -0.636936},
                                    'rpy': {'p': -0.737654, 'r': -0.618076, 'y': -1.33747},
                                    't': {'x': 0.092225, 'y': -0.008196, 'z': 1.013705}},
                            'ts': 0.017647},
                      'cam': {'pos': {'q': {'w': 1.0, 'x': 6.1e-05, 'y': 0.000162, 'z': 0.000107},
                                      'rpy': {'p': 0.000324, 'r': 0.000122, 'y': 0.000214},
                                      't': {'x': -6.6e-05, 'y': -9e-05, 'z': 8e-06}},
                              'ts': 0.017647},
                      'gt_frame': 'depth_mask_18446744073709551615.png',
                      'id': 18446744073709551615,
                      'ts': 0.017647}],
 'imu': {'/prophesee/left/imu': [{'angular_velocity': {'x': -0.001065, 'y': 0.011984, 'z': 0.020639},
                                  'linear_acceleration': {'x': 8.293953, 'y': 0.349074, 'z': -5.420528},
                                  'ts': 0.017303}],
         '/prophesee/right/imu': [{'angular_velocity': {'x': -0.036885, 'y': 0.021438, 'z': -0.006924},
                                   'linear_acceleration': {'x': -8.361612, 'y': 0.078437, 'z': -4.883445},
                                   'ts': 0.017319}]},
 'meta': {'cx': 1053.709961,
          'cy': 788.531982,
          'dist_model': 'radtan',
          'fx': 2066.48999,
          'fy': 2066.469971,
          'k1': -0.117189,
          'k2': 0.069793,
          'k3': 0.0,
          'k4': 0.0,
          'p1': -0.000327,
          'p2': 0.005909,
          'res_x': 2080,
          'res_y': 1552,
          'ros_time_offset': 1606265321.445115}}
```

## Transform Convention
- Object poses represent transforms from the object frame to the camera frame.
- Camera poses represent transforms from the camera frame to the world frame.


## Format Comparison
### EVIMO2v1 vs EVIMO2v2
Many important improvements to the data generation pipeline that transforms raw recordings to released data were made in order to release EVIMO2v2. These changes include:

- All camera's data's timestamps are synchronized to a common source (Vicon)
- Event camera ground truths are synchronized and jitter is eliminated up to numercial precision
- Classical camera's ground truth jitter is eliminated up to numerical precision
- Classical camera's image data is kept when there is no ground truth due to Vicon occlusion
- A redesigned NPZ format greatly reduces loading time without requiring excessive disk usage
- Events are not filtered when copied out of the `raw` recordings
- Several sequences with no usable data were deleted
- Sequences were split into several sub-sequences satisfying:
    - Gaps in ground truth depth/mask have a duration of at most 1 second
    - All sub-sequences are at least 0.4 seconds long

### EVIMO2v2 TXT vs EVIMO2v2 NPZ

Decompressed, the EVIMO2v2 NPZ format requires 525 GB of space of which 271 GB is for the RGB camera and 255 GB is for the three DVS cameras.

Decompressed, the TXT format requires 842 GB of space of which 212 GB is for the RGB camera and 631 GB is for the three DVS cameras.

The EVIMO2v2 NPZ format supports memory mapping the event arrays from disk. 

The TXT format requires parsing the `events.txt` file, which can tens of seconds per sequence.

The TXT format decompresses to about 471,000 files. The npz format decompresses to 4,700 files.

### EVIMO, EVIMO2v1 NPZ vs EVIMO2v2 NPZ

EVIMO and EVIMO2v1 compressed the entire dataset into a single `.npz` file. Inside this `.npz` there are archives for the depth, masks, events, and conventional images. To access any of this data the entire depth, mask, events, or conventional archives must be decompressed into RAM (by numpy) or onto disk (manually by the user). This can take several minutes for each sequence and requires the user to either have upwards of 64GB of RAM (to hold a decompressed sequence) or multiple TB of disk space (to store all sequences decompressed at once).

EVIMO2v2 introduces a new NPZ format with two major changes. These changes eliminate long sequence load times and while preventing excessive disk usage. First, depth, masks, and conventional images are compressed frame by frame so that only the frame being currently used needs to be decompressed and stored in RAM. Second, events are stored uncompressed with the minimum width data type. Because the events are stored in a decompressed format, the arrays can be memory mapped so that only the portions of the events currently in use need to be stored in RAM.
