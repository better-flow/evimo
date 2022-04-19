# Ground Truth Format

The ground truth is available in 2 formats:
 1. `.npz` format: ground truth is saved as a single `.npz`, which is convenient to use in Python.
 2. Text file format: ground truth and recording is saved in the form of `.png` images and `.txt` files.

We strongly suggest using the [pydvs sample](https://github.com/better-flow/pydvs/blob/bdff8de0c3c7df24d3143154b415091d88a1e4c2/samples/evimo-gen.py) as a prototype for manipulating the data (in both formats). This is the script which converts the raw text format (the output of the [C++ pipeline](https://github.com/better-flow/evimo/wiki/Ground-Truth-Generation)) to `.npz`. The script also performs typical manipulations with the data, such as plotting the trajectories, generating event slices and overlaying mask on the top of camera frames or event slices.

## NPZ (EVIMO2v2)

## NPZ (EVIMO1 and EVIMO2v1)
This format consists of a single `.npz` file:

<img src="https://github.com/better-flow/evimo/blob/master/docs/wiki_img/npz_format.png" width="500"/>

- `depth.npy` / `mask.npy` - an array of shape *(NUM_FRAMES, RES_Y, RES_X)* with ground truth; depth is in *mm.*, mask has object ids multiplied by `1000`.
- `classical.npy` : an array of shape *(NUM_FRAMES, RES_Y, RES_X)* **OR** `None`. Contains classical camera when available.
- `K` / `D` - intrinsic and distortion parameters. The camera model type needs to be read from `meta`, and we suggest reading intrinsic parameters from `meta` as well.
- `meta` - a python dict saved as `.npy`. The contents are described [here](https://github.com/better-flow/evimo/wiki/Ground-Truth-Generation); See [example](https://github.com/better-flow/pydvs/blob/bdff8de0c3c7df24d3143154b415091d88a1e4c2/samples/evimo-gen.py#L217) on how to access and plot the data.
- `events` - an array of shape *(NUM_EVENTS, 4)*; contains timestamps, x/y pixel coordinates and polarity per event. Getting a slice using `pydvs` functionality ([link](https://github.com/better-flow/pydvs/blob/bdff8de0c3c7df24d3143154b415091d88a1e4c2/samples/evimo-gen.py#L282)) and implementation in `pydvs` ([link](https://github.com/better-flow/pydvs/blob/bdff8de0c3c7df24d3143154b415091d88a1e4c2/lib/pydvs.py#L275)). **Note**: `pydvs` implementation of the slice is approximate; the timestamp boundaries of events may not be the closest to the ones specified.
- `index` - a helper lookup table for fast timestamp-to-event index computation. Contains indices of events every `discretization` seconds. See how it is generated [here](https://github.com/better-flow/pydvs/blob/bdff8de0c3c7df24d3143154b415091d88a1e4c2/lib/pydvs.py#L123) and a simple way to extract an (approximate) event slice [here](https://github.com/better-flow/pydvs/blob/bdff8de0c3c7df24d3143154b415091d88a1e4c2/samples/error_demo.py#L200) - here the width of event slice is 1 x `discretization`.

## TXT
This is the direct result of the data generation in the C++ pipeline (with the exception of the trajectory plot) and is used to generate `.npz` later on. The contents and format of individual files is described [here](https://github.com/better-flow/evimo/wiki/Ground-Truth-Generation), and camera calibration files are covered [here](https://github.com/better-flow/evimo/wiki/Dataset-Configuration-Folder).

The packaged dataset will look similar to:

<img src="https://github.com/better-flow/evimo/blob/master/docs/wiki_img/text_format.png" width="800"/>

- `img`: a folder with depth images, masks (within the same image in separate channels) and optionally classical camera frames.
- `calib.txt`, `extrinsics.txt` and `params.txt` (or `config.txt`) are camera parameters.
- `events.txt`: file with events (if available for a given camera).
- `meta.txt`: file with Vicon tracks (it also contains camera calibration).
- `position_plots.pdf`: for visualization only, a plot of camera/object trajectories.


## Format comparisons
### TXT vs NPZ (EVIMO2 v2)

Decompressed, the NPZ format requires 525 GB of space of which 271 GB is for the RGB camera and 255 GB is for the three DVS cameras.

Decompressed, the TXT format requires 842 GB of space of which 212 GB is for the RGB camera and 631 GB is for the three DVS cameras.

The NPZ format supports memory mapping the event arrays from disk. 

The TXT format requires parsing the `events.txt` file, which can tens of seconds per sequence.

The TXT format decompresses to about 471,000 files. The npz format decompresses to 4700 files.

### EVIMO, EVIMO2v1 NPZ vs EVIMO2v2 NPZ

EVIMO and EVIMO2v1 compressed the entire dataset into a single `.npz` file. Inside this `.npz` there are archives for the depth, masks, events, and conventional images. To access any of this data the entire depth, mask, events, or conventional archives must be decompressed into RAM (by numpy) or onto disk (manually by the user). This can take several minutes for each sequence and requires the user to either have upwards of 64GB of RAM (to hold a decompressed sequence) or multiple TB of disk space (to store all sequences decompressed at once).

EVIMO2v2 introduces a new NPZ format with two major changes. These changes eliminate long sequence load times and while preventing excessive disk usage. First, depth, masks, and conventional images are compressed frame by frame so that only the frame being currently used needs to be decompressed and stored in RAM. Second, events are stored uncompressed with the minimum width data type. Because the events are stored in a decompressed format, the arrays can be memory mapped so that only the portions of the events currently in use need to be stored in RAM.
