# Ground Truth Format

The ground truth is available in the following formats:

* `NPZ`
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
|`dataset_mask.npz`| Dictionary of *(RES_Y, RES_X)* arrays with keys `mask_<frame id>`|
|`dataset_events_p.npy`| Array of *(NUM_EVENTS, 1) containing events polarity <br> **Can be memory mapped**|
|`dataset_events_t.npy`| Array of *(NUM_EVENTS, 1) containing events time<br> **Can be memory mapped**|
|`dataset_events_xy.npy`| Array of *(NUM_EVENTS, 2) containg events pixel location<br> **Can be memory mapped**|
|`dataset_info.npz`| Dictionary of arrays `D`, `discretization`, `index`, `K`, `meta`<br> Contents are identical to the EVIMO, EVIMO2v1 NPZ format|

## NPZ (EVIMO and EVIMO2v1)
There is one compressed `.npz` file per sequence. A sequences file can be found in the following path: `<camera>/<category>/<subcategory>/<sequence name>.npz`

A sequences `.npz` file contains the following `.npy` files:

|File |Description |
--- | --- |
|`classical.npy`| Array with shape *(NUM_FRAMES, RES_Y, RES_X)*<br> Contains classical frames if available|
|`depth.npy`| Array of shape *(NUM_FRAMES, RES_Y, RES_X)*<br>depth is in *mm*|
|`mask.npy` | Array of shape *(NUM_FRAMES, RES_Y, RES_X)*<br>masks contains object ids multiplied by `1000`|
|`events.npy`| Array of shape *(NUM_EVENTS, 4)*<br>Each row contains an events timestamp, x/y pixel coordinate and polarity
|`meta.npy`| Python dictionary containing intrinsics, timestamps, poses, and IMU data<br>The full description is [here](ground-truth-format.md#metas-details)
|`K.npy/D.npy`| Intrinsic and distortion parameters, also available in `meta.npy`|
|`index.npy`| A helper lookup table for fast timestamp-to-event index computation<br>Contains indices of events every `discretization.npy` seconds.
|`discretization.npy`| The time between events corresponding to the indices in `index.npy`|

## TXT
There is one folder per sequence. A sequences folder can be found in the following path:<br>
`<camera>/<category>/<subcategory>/<sequence name>`

We strongly suggest using the [pydvs sample](https://github.com/better-flow/pydvs/blob/bdff8de0c3c7df24d3143154b415091d88a1e4c2/samples/evimo-gen.py) as a prototype for manipulating the TXT data. This is the script which converts the TXT format (the output of the [C++ pipeline](offline-generation-tool.md) to NPZ. It also generates the trajectory plot distributed with the TXT format, event slices, and frames for visualization videos.

|Item |Description |
--- | --- |
|`img/img_<frame id>.png`| Conventional frames from a classical camera, when available|
|`img/depth_mask_<frame id>.png`| 16-bit png's with depth and masks in different channels <br>Depth is in *mm*.<br> Masks contain per-pixel object ids multiplied by `1000`.<br>A Python example on how to read the image can be found here](https://github.com/better-flow/pydvs/blob/bdff8de0c3c7df24d3143154b415091d88a1e4c2/samples/evimo-gen.py#L229) and [here](https://github.com/better-flow/pydvs/blob/bdff8de0c3c7df24d3143154b415091d88a1e4c2/samples/evimo-gen.py#L11).<br>The relevant C++ code is [here](https://github.com/better-flow/evimo/blob/3b20a8a3ee729b9b9eb69bda05ac4cbf8b9773cb/evimo/dataset_frame.h#L300).|
|`events.txt`| File with events (if available)<br>One event per line in a format `<timestamp px py polarity>`|
|`calib.txt`<br>`extrinsics.txt`<br> `params.txt` or `config.txt`| Camera parameters see [here](dataset-configuration-folder.md#camera-configuration-folder-example) for details|
|`meta.txt`| String containing intrinsics, timestamps, poses, and IMU data<br>The full description is [here](ground-truth-format.md#metas-details)|
|`position_plots.pdf`| A plot of camera/object trajectories for visualization only|


## Meta's Contents

An example of the contents of `meta.npy` or `meta.txt` is given below. In either case, the structure is a Python dictionary.
```
{'frames': [{'22': {'pos': {'q': {'w': 0.854428, 'x': 0.517475, 'y': 0.046191, 'z': -0.006281},
                            'rpy': {'p': 0.085538, 'r': 1.090704, 'y': 0.037217},
                            't': {'x': -0.389233, 'y': -0.082026, 'z': 0.763271}},
                    'ts': 5.446962},
             '9': {'pos': {'q': {'w': 0.88787, 'x': -0.109701, 'y': -0.251739, 'z': -0.36916},
                           'rpy': {'p': -0.556265, 'r': -0.010523, 'y': -0.785069},
                           't': {'x': -0.493914, 'y': 0.004656, 'z': 0.491391}},
                   'ts': 5.446962},
             'cam': {'pos': {'q': {'w': 0.991128, 'x': -0.045218, 'y': 0.070844, 'z': 0.102964},
                             'rpy': {'p': 0.150307, 'r': -0.075974, 'y': 0.201305},
                             't': {'x': 0.463163, 'y': -0.075241, 'z': 0.107412}},
                     'ts': 5.446962,
                     'vel': {'q': {'w': 0.925566, 'x': -0.068081, 'y': -0.34568, 'z': -0.138557},
                             'rpy': {'p': -0.719177, 'r': -0.040196, 'y': -0.282081},
                             't': {'x': -0.550138, 'y': 0.452351, 'z': 0.056179}}},
             'gt_frame': 'depth_mask_18446744073709551615.png',
             'id': 18446744073709551615,
             'ts': 5.446962}],
 'meta': {'cx': 328.38266,
          'cy': 245.452011,
          'dist_model': 'radtan',
          'fx': 519.299927,
          'fy': 515.430115,
          'k1': 0.091035,
          'k2': -0.118269,
          'k3': 0.0,
          'k4': 0.0,
          'p1': 0.004237,
          'p2': 0.004445,
          'res_x': 640,
          'res_y': 480}}
```

### Meta's Details
**TODO** Describe every key and make this a table

- A special object id `'cam'` is used for camera
- The `'gt_frame'` key contains the `depth_mask_<frame id>.png` file in the TXT format
- The `'classical_frame'` key contains the `img_<frame id>.png` file in the TXT format
- The `'id'` key can be used to find the `img_<frame id>.png` file in the TXT format
- The `'id'` key can be used to generate the `classical_<frame id>`, `depth_<frame id>`, `mask_<frame id>` keys used in the EVIMO2v2 NPZ format
- **Note:** the `meta.txt` contains two duplicates of the Vicon trajectory: one with ground truth (the key is `'frames'`) - generated at the frame rate of ground truth. Another - `'full_trajectory'` has Vicon messages at full rate (200Hz), but keys `'id'` and `'gt_frame'` cannot be used.
- The `'vel'` key is deprecated and should be ignored
- IMU data is available under **TODO**
- `meta.txt` is a Python dictionary that can be parsed with the Python `eval()` method. See [here](https://github.com/better-flow/pydvs/blob/bdff8de0c3c7df24d3143154b415091d88a1e4c2/samples/evimo-gen.py#L182).

## Format Comparison
### EVIMO2v2 TXT vs EVIMO2v2 NPZ

Decompressed, the EVIMO2v2 NPZ format requires 525 GB of space of which 271 GB is for the RGB camera and 255 GB is for the three DVS cameras.

Decompressed, the TXT format requires 842 GB of space of which 212 GB is for the RGB camera and 631 GB is for the three DVS cameras.

The EVIMO2v2 NPZ format supports memory mapping the event arrays from disk. 

The TXT format requires parsing the `events.txt` file, which can tens of seconds per sequence.

The TXT format decompresses to about 471,000 files. The npz format decompresses to 4,700 files.

### EVIMO, EVIMO2v1 NPZ vs EVIMO2v2 NPZ

EVIMO and EVIMO2v1 compressed the entire dataset into a single `.npz` file. Inside this `.npz` there are archives for the depth, masks, events, and conventional images. To access any of this data the entire depth, mask, events, or conventional archives must be decompressed into RAM (by numpy) or onto disk (manually by the user). This can take several minutes for each sequence and requires the user to either have upwards of 64GB of RAM (to hold a decompressed sequence) or multiple TB of disk space (to store all sequences decompressed at once).

EVIMO2v2 introduces a new NPZ format with two major changes. These changes eliminate long sequence load times and while preventing excessive disk usage. First, depth, masks, and conventional images are compressed frame by frame so that only the frame being currently used needs to be decompressed and stored in RAM. Second, events are stored uncompressed with the minimum width data type. Because the events are stored in a decompressed format, the arrays can be memory mapped so that only the portions of the events currently in use need to be stored in RAM.
