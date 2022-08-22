#!python
#
# ###############################################################################
#
# File: evimo_flow.py
#
# Calculate optical flow from EVIMO datasets
# so far this is tested on EVIMO2 for VGA DVS reocrdings
#
# usage: evimo_flow.py [-h] [--dt DT] [--quiet] [--overwrite] [--wait] [--dframes DFRAMES] [files ...]
#
# positional arguments:
# files              NPZ files to convert
#
# optional arguments:
# -h, --help         show this help message and exit
# --dt DT            dt for flow approximation"dt" is how far ahead of the camera trajectory to sample in secondswhen approximating flow through finite
# difference. Smaller values are more accurate, but noiser approximations of optical flow.
# --quiet            turns off OpenCV graphical output windows
# --overwrite        Overwrite existing output files
# --wait             Wait for keypress between visualizations (for debugging)
# --dframes DFRAMES  Alternative to flow_dt, flow is calculated for time N depth frames ahead
# --format format    Either "evimo2v1" or "evimo2v2" which are the different input NPZ formats supported
#
# Calculates optical flow from EVIMO datasets. Each of the source npz files on the command line is processed to produce the corresponding flow npz files. See source
# code for details of output.
# The source evimo_file.npz file(s) are one of the EV-IMO NPZ files that combine all the sensor and GT static pose data, e.g.
# samsung_mono/imo/train/scene9_dyn_train_02.npz
# The source NPZ file contents are documented in https://github.com/better-flow/evimo/wiki/Ground-Truth-Format
#
# "DT" is how far ahead of the camera trajectory to sample in seconds
# when approximating flow through finite difference. Smaller values are
# more accurate, but noiser approximations of optical flow.
#
# "DFRAMES" is  useful because the resulting displacement arrows point to the new position of points in the scene at the time of a ground truth frame in the future.
# The displacements are correct for even for insane values, like 10, or 20 frames ahead from the current gt_frame. Combined with the --wait flag, we use dframes to make sure everything is working correctly.
# Here is a convincing example:
#      python3 evimo_flow.py --overwrite ../../recordings/samsung_mono/imo/eval/scene15_dyn_test_05.npz --dframe 3 --wait
#
# Press the a button until the board starts to flip the toys, then use your mouse to verify the toys move where the arrows say they will, even though the arrows are computed "far" into the future.
#
# Writes out flow files to the NPZ file folder location as:
# evimo_file_flow.npz
# which contains:
# timestamps.npy
# end_timestamps.npy
# x_flow_dist.npy
# y_flow_dist.npy
#
# timestamps are relative to epoch time in double seconds
# end_timestamps are relative to epoch_time in double seconds, they correspond to
# the end of the interval x_flow and y_flow were computed from (e.g with default settings, 0.01 seconds ahead of timestamps)
# x_flow and y_flow are displacements in pixels over the time period timestamp to end_timestamp.
# To obtain flow speed in px/s, divide displacement dx,dy by the time difference (end_timestamp-timestamp).
#
# Missing depth values (walls of room) are filled with NaN
# The timestamps can skip values when the VICON has lost lock on one or more objects from occulusion or too rapid motion.
#
#
# To use this converter standalone, make a python 3 environment and in it
# pip install argparse numpy scipy opencv-python easygui tqdm
#
# History:
# 01-21-22 - Levi Burner - Created file
# 01-29-22 - Tobi Delbruck - corrected to match MVSEC NPZ flow output contents for mono camera recordings from MVSEC, added multiple file option
# 01-30-22 - Levi Burner - Fixed to handle moving objects correctly and fixed bug in radtan model
# 04-26-22 - Levi Burner - Support EVIMO2v2 format
# 08-23-22 - Levi Burner - Add classical camera reprojection for EVIMO2v2 format
#
###############################################################################

import argparse
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' # Attempt to disable OpenBLAS multithreading used by NumPy, it makes the script 17% slower
import multiprocessing
from multiprocessing import Pool
import zipfile
import cv2
import numpy as np
import pprint
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import easygui
from tqdm import tqdm
from pathlib import Path
import statistics

# See NumPy source code to see how these functions are normally used to create NPZ files
def create_compressed_npz(file, compresslevel=None):
    npz = np.lib.npyio.zipfile_factory(file, mode='w', compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel)
    return npz

def add_to_npz(npz, name, array):
    with npz.open(name + '.npy', 'w', force_zip64=True) as f:
        np.lib.format.write_array(f, array, allow_pickle=False, pickle_kwargs=None)

def close_npz(npz): npz.close()

# https://stackoverflow.com/a/57364423
# istarmap.py for Python 3.8+
import multiprocessing.pool as mpp
def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)
mpp.Pool.istarmap = istarmap

# Sample a list of translations and rotations
# with linear interpolation
def interpolate_pose(t, pose):
    right_i = np.searchsorted(pose[:, 0], t)
    if right_i==pose.shape[0]:
        # print(f'attempted extrapolation past end of poses, clipping')
        right_i=right_i-1 # prevent attempted extrapolation past array end

    if right_i==0:
        # print(f'attempt extrapolation past beginning of poses, clipping')
        right_i=1 # prevent attempted extrapolation before array start

    left_t  = pose[right_i-1, 0]
    right_t = pose[right_i,   0]

    alpha = (t - left_t) / (right_t - left_t)
    if alpha>1:
        # print(f'attempted alpha>1, clipping')
        alpha=1
    elif alpha < 0:
        # print(f'attempted alpha<0, clipping')
        alpha=0

    left_position  = pose[right_i - 1, 1:4]
    right_position = pose[right_i,     1:4]

    position_interp = alpha * (right_position - left_position) + left_position

    left_right_rot_stack = R.from_quat((pose[right_i - 1, 4:8],
                                        pose[right_i,     4:8]))

    slerp = Slerp((0, 1), left_right_rot_stack)
    R_interp = slerp(alpha)

    return np.array([t,] + list(position_interp) + list(R_interp.as_quat()))

# Apply an SE(3) transform to an element of SE(3)
# e.g. T_cb to T_ba to get T_ca
# Transforms are the same thing as poses
def apply_transform(T_cb, T_ba):
    R_ba = R.from_quat(T_ba[4:8])
    t_ba = T_ba[1:4]

    R_cb = R.from_quat(T_cb[4:8])
    t_cb = T_cb[1:4]

    R_ca = R_cb * R_ba
    t_ca = R_cb.as_matrix() @ t_ba + t_cb
    return np.array([T_ba[0],] + list(t_ca) + list(R_ca.as_quat()))

# Invert an SE(3) transform
def inv_transform(T_ba):
    R_ba = R.from_quat(T_ba[4:8])
    t_ba = T_ba[1:4]

    R_ab = R_ba.inv()
    t_ab = -R_ba.inv().as_matrix() @ t_ba

    return np.array([T_ba[0],] + list(t_ab) + list(R_ab.as_quat()))

# OpenCV's map functions do not return results for fractional
# start points. Best to implement by hand.
# https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a
# https://github.com/better-flow/evimo/blob/967d4d688ecee08e0c66b0c86c991980434e6c41/evimo/dataset.h#L662
def project_points_radtan(points,
                          fx, fy, cx, cy,
                          k1, k2, p1, p2):
    x_ = np.divide(points[:, :, 0], points[:, :, 2], out=np.zeros_like(points[:, :, 0]), where=points[:, :, 2]!=0)
    y_ = np.divide(points[:, :, 1], points[:, :, 2], out=np.zeros_like(points[:, :, 1]), where=points[:, :, 2]!=0)

    r2 = np.square(x_) + np.square(y_)
    r4 = np.square(r2)

    dist = (1.0 + k1 * r2 + k2 * r4)

    x__ = x_ * dist + 2.0 * p1 * x_ * y_ + p2 * (r2 + 2.0 * x_ * x_)
    y__ = y_ * dist + 2.0 * p2 * x_ * y_ + p1 * (r2 + 2.0 * y_ * y_)


    u = fx * x__ + cx
    v = fy * y__ + cy

    return u, v

# Generate an HSV image using color to represent the gradient direction in a optical flow field
def visualize_optical_flow(flowin):
    flow=np.ma.array(flowin, mask=np.isnan(flowin))
    theta = np.mod(np.arctan2(flow[:, :, 0], flow[:, :, 1]) + 2*np.pi, 2*np.pi)

    flow_norms = np.linalg.norm(flow, axis=2)
    flow_norms_normalized = flow_norms / np.max(np.ma.array(flow_norms, mask=np.isnan(flow_norms)))

    flow_hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flow_hsv[:, :, 0] = 179 * theta / (2*np.pi)
    flow_hsv[:, :, 1] = 255 * flow_norms_normalized
    flow_hsv[:, :, 2] = 255 * (flow_norms_normalized > 0)

    return flow_hsv

def draw_flow_arrows(img, xx, yy, dx, dy, p_skip=15, mag_scale=1.0):
    xx     = xx[::p_skip, ::p_skip].flatten()
    yy     = yy[::p_skip, ::p_skip].flatten()
    flow_x = dx[::p_skip, ::p_skip].flatten()
    flow_y = dy[::p_skip, ::p_skip].flatten()

    for x, y, u, v in zip(xx, yy, flow_x, flow_y):
        if np.isnan(u) or np.isnan(v):
            continue
        cv2.arrowedLine(img,
                        (int(x), int(y)),
                        (int(x+mag_scale*u), int(y+mag_scale*v)),
                        (0, 0, 0),
                        tipLength=0.2)

def flow_direction_image(shape=(60,60)):
    (x, y) = np.meshgrid(np.arange(0, shape[1]) - shape[1]/2,
                         np.arange(0, shape[0]) - shape[0]/2)
    theta = np.mod(np.arctan2(x, y) + 2*np.pi, 2*np.pi)

    flow_hsv = np.zeros((shape[0], shape[1], 3)).astype(np.uint8)
    flow_hsv[:, :, 0] = 179 * theta / (2*np.pi)
    flow_hsv[:, :, 1] = 255
    flow_hsv[:, :, 2] = 255
    return flow_hsv

# Get a poses of all objects in the vicon frame
# out of EVIMO's 'meta' field and into a numpy array
def get_all_poses(meta):
    vicon_pose_samples = len(meta['full_trajectory'])

    poses = {}
    key_i = {}
    for key in meta['full_trajectory'][0].keys():
        if key == 'id' or key == 'ts' or key == 'gt_frame':
            continue
        poses[key] = np.zeros((vicon_pose_samples, 1+3+4))
        key_i[key] = 0

    # Convert camera poses to array
    for all_pose in meta['full_trajectory']:
        for key in poses.keys():
            if key == 'id' or key == 'ts' or key == 'gt_frame':
                continue

            if key in all_pose:
                i = key_i[key]
                poses[key][i, 0] = all_pose['ts']
                poses[key][i, 1] = all_pose[key]['pos']['t']['x']
                poses[key][i, 2] = all_pose[key]['pos']['t']['y']
                poses[key][i, 3] = all_pose[key]['pos']['t']['z']
                poses[key][i, 4] = all_pose[key]['pos']['q']['x']
                poses[key][i, 5] = all_pose[key]['pos']['q']['y']
                poses[key][i, 6] = all_pose[key]['pos']['q']['z']
                poses[key][i, 7] = all_pose[key]['pos']['q']['w']
                key_i[key] += 1

    for key in poses.keys():
        poses[key] = poses[key][:key_i[key], :]

    return poses

# Get the camera intrinsics out of EVIMO's 'meta' field
def get_intrinsics(meta):
    meta_meta = meta['meta']
    K = np.array(((meta_meta['fx'],          0, meta_meta['cx']),
                  (         0, meta_meta['fy'], meta_meta['cy']),
                  (         0,          0,          1)))

    dist_coeffs = np.array((meta_meta['k1'],
                            meta_meta['k2'],
                            meta_meta['p1'],
                            meta_meta['p2']))

    return K, dist_coeffs

def load_data(file, format='evimo2v2'):
    if format == 'evimo2v1':
        data = np.load(file, allow_pickle=True)
        meta  = data['meta'].item()
        depth = data['depth']
        mask  = data['mask']
    elif format == 'evimo2v2':
        meta  = np.load(os.path.join(file, 'dataset_info.npz'), allow_pickle=True)['meta'].item()
        depth = np.load(os.path.join(file, 'dataset_depth.npz'))
        mask  = np.load(os.path.join(file, 'dataset_mask.npz'))
    else:
        raise Exception('Unrecognized data format')
    return meta, depth, mask

def load_data_bgr(file, format='evimo2v2'):
    assert format == 'evimo2v2'
    meta  = np.load(os.path.join(file, 'dataset_info.npz'), allow_pickle=True)['meta'].item()
    depth = np.load(os.path.join(file, 'dataset_depth.npz'))
    bgr = np.load(os.path.join(file, 'dataset_classical.npz'))    
    return meta, depth, bgr

# Only for evimo2v2
def load_extrinsics(file):
    ext = np.load(os.path.join(file, 'dataset_extrinsics.npz'), allow_pickle=True)
    q = ext['q_rigcamera'].item()
    q_rc = np.array([q['x'], q['y'], q['z'], q['w']])
    t = ext['t_rigcamera'].item()
    t_rc = np.array([t['x'], t['y'], t['z']])
    T_rc = np.array([0, *t_rc, *q_rc])
    return T_rc

def convert(file, flow_dt, quiet=False, showflow=True, overwrite=False,
            waitKey=1, dframes=None, format='evimo2v1', bgr_file=None):
    cv2.setNumThreads(1) # OpenCV is wasteful and so we run this with one process per sequence

    if format != 'evimo2v2' and bgr_file is not None:
        raise ValueError('reproject_bgr option only supported for evimo2v2 format due to high computational overhead of older formats')

    if file.endswith('_flow.npz'):
        if not quiet: print(f'skipping {file} because it appears to be a flow output npz file')
        return

    if format == 'evimo2v1':
        npz_name_base=file[:-4] + '_flow'
        npz_flow_file_name = npz_name_base+'.npz'
    elif format == 'evimo2v2':
        npz_name_base = file
        npz_flow_file_name = os.path.join(file, 'dataset_flow.npz')
        npz_bgr_file_name = os.path.join(file, 'dataset_reprojected_classical.npz')
    else:
        raise Exception('Unsupported EVIMO format')

    if not overwrite and os.path.exists(npz_flow_file_name):
        print(f'skipping {file} because {npz_flow_file_name} exists; use --overwrite option to overwrite existing output file')
        return

    if dframes is None:
        if not quiet: print(f'converting {file} with dt={flow_dt}s; loading source...',end='')
    else:
        if not quiet: print(f'converting {file} with dframes={dframes}; loading source...',end='')

    meta, depth, mask = load_data(file, format=format)

    if bgr_file is not None:
        bgr_meta, bgr_depth, bgr_bgr = load_data_bgr(bgr_file, format=format)

    if not quiet: print('done loading')

    if format == 'evimo2v1':
        depth_shape     = depth[0]    .shape
        bgr_depth_shape = bgr_depth[0].shape
    elif format == 'evimo2v2':
        depth_shape     = depth[next(iter(depth))]    .shape
        if bgr_file is not None: bgr_depth_shape = bgr_depth[next(iter(bgr_depth))].shape
    else:
        raise Exception('Unsupport data format')

    all_poses      = get_all_poses (meta)
    K, dist_coeffs = get_intrinsics(meta)

    if bgr_file is not None:
        bgr_K, bgr_dist_coeffs = get_intrinsics(bgr_meta)
        T_gr = load_extrinsics(bgr_file)
        T_gc = load_extrinsics(file)
        T_cr = apply_transform(inv_transform(T_gc), T_gr)

    # Get the map from pixels to direction vectors with Z = 1
    map1, map2 = cv2.initInverseRectificationMap(
        K, # Intrinsics
        dist_coeffs, # Distortion
        np.eye(3), # Rectification
        np.eye(3), # New intrinsics
        (depth_shape[1], depth_shape[0]),
        cv2.CV_32FC1)

    # Initial positions of every point
    x = np.arange(0, depth_shape[1], 1)
    y = np.arange(0, depth_shape[0], 1)
    xx, yy = np.meshgrid(x, y)
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    if showflow:
        # For visualization
        flow_direction_image_hsv = flow_direction_image((depth_shape[0], depth_shape[1]))
        flow_direction_image_bgr = cv2.cvtColor(flow_direction_image_hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('color direction chart', flow_direction_image_bgr)

    # Find frames with depth and masks
    # Getting the list of names in one go is a big speedup because it avoids
    # looking into the NPZ file over and over
    depth_names = list(depth.keys())
    mask_names  = list(mask .keys())
    frames_with_depth = []
    frames_with_depth_t = []
    for i, frame_info in enumerate(meta['frames']):
        if format == 'evimo2v1':
            frames_with_depth.append(frame_info)
        elif format == 'evimo2v2':
            first_frame_id = meta['frames'][0]['id']
            depth_frame_mm_key = 'depth_' + str(frame_info['id'] - first_frame_id).rjust(10, '0')
            mask_frame_key     = 'mask_'  + str(frame_info['id'] - first_frame_id).rjust(10, '0')
            if depth_frame_mm_key in depth_names and mask_frame_key in mask_names:
                frames_with_depth.append(frame_info)
                frames_with_depth_t.append(frame_info['ts'])
        else:
            raise Exception('Unsupport EVIMO format')
    frames_with_depth_t = np.array(frames_with_depth_t)

    if bgr_file is not None:
        bgr_depth_names = list(bgr_depth.keys())
        bgr_bgr_names   = list(bgr_bgr  .keys())
        bgr_frames_with_depth = []
        bgr_frames_with_depth_t = []
        for frame_info in bgr_meta['frames']:
            first_frame_id = bgr_meta['frames'][0]['id']
            depth_frame_mm_key = 'depth_'    + str(frame_info['id'] - first_frame_id).rjust(10, '0')
            bgr_frame_key      = 'classical_'+ str(frame_info['id'] - first_frame_id).rjust(10, '0')
            if depth_frame_mm_key in bgr_depth_names and bgr_frame_key in bgr_bgr_names:
                bgr_frames_with_depth.append(frame_info)
                bgr_frames_with_depth_t.append(frame_info['ts'])
        bgr_frames_with_depth_t = np.array(bgr_frames_with_depth_t)

        # Select bgr frames just after the event camera depth frames
        bgr_frames_indices = np.searchsorted(bgr_frames_with_depth_t, frames_with_depth_t)

        # Could also filter if dt is too large here
        bgr_frames_for_event_frames_with_depth = []
        for i in bgr_frames_indices:
            if i == len(bgr_frames_with_depth):
                i = len(bgr_frames_with_depth) - 1
            if i < 0:
                i = 0
            bgr_frames_for_event_frames_with_depth.append(bgr_frames_with_depth[i])
    else:
        bgr_frames_for_event_frames_with_depth = [None]*len(frames_with_depth)

    if dframes is not None:
        iterskip = dframes
    else:
        iterskip = 1

    # calculate number of output frames
    num_output_frames = int((len(frames_with_depth)-1) / iterskip) + 1

    # Preallocate arrays for flow as in MSEVC format
    if format == 'evimo2v1':
        timestamps      = np.zeros((num_output_frames,), dtype=np.float64)
        end_timestamps  = np.zeros((num_output_frames,), dtype=np.float64)
        flow_shape = (num_output_frames, *depth_shape)
        x_flow_dist = np.zeros(flow_shape, dtype=np.float64) # named as in MVSEC monoocular camera flow, double as in MVSEC NPZs
        y_flow_dist = np.zeros(flow_shape, dtype=np.float64)
    elif format == 'evimo2v2':
        timestamps      = np.zeros((num_output_frames,), dtype=np.float64)
        end_timestamps  = np.zeros((num_output_frames,), dtype=np.float64)
        npz_flow_file = create_compressed_npz(npz_flow_file_name)
        if bgr_file is not None:
            npz_bgr_file = create_compressed_npz(npz_bgr_file_name)
    else:
        raise Exception('Unsupported EVIMO format')

    ros_time_offset=meta['meta']['ros_time_offset'] # get offset of start of data in epoch seconds to write the flow timestamps using same timebase as MVSEC
    last_time=float('nan')
    last_delta_time=float('nan')
    delta_times=[]
    start_time=float('nan')
    large_delta_times=0

    for i, (frame_info, bgr_frame_info) in enumerate(
        tqdm(list(zip(frames_with_depth[::iterskip],
                      bgr_frames_for_event_frames_with_depth[::iterskip])),
             position=1, desc='{}'.format(os.path.split(file)[1]))):
        if format == 'evimo2v1':
            depth_frame_mm_key = iterskip*i
            mask_frame_key     = iterskip*i
            depth_frame_mm = depth[depth_frame_mm_key]
            mask_frame     = mask[mask_frame_key]
        elif format == 'evimo2v2':
            first_frame_id = meta['frames'][0]['id']
            depth_frame_mm_key = 'depth_' + str(frame_info['id'] - first_frame_id).rjust(10, '0')
            mask_frame_key     = 'mask_'  + str(frame_info['id'] - first_frame_id).rjust(10, '0')
            depth_frame_mm = depth[depth_frame_mm_key]
            mask_frame = mask [mask_frame_key]
        else:
            raise Exception('Unsupport EVIMO format')

        if bgr_file is not None:
            bgr_first_frame_id = bgr_meta['frames'][0]['id']
            bgr_depth_frame_mm_key = 'depth_'      + str(bgr_frame_info['id'] - bgr_first_frame_id).rjust(10, '0')
            bgr_bgr_frame_key      = 'classical_'  + str(bgr_frame_info['id'] - bgr_first_frame_id).rjust(10, '0')
            bgr_depth_frame_mm = bgr_depth[bgr_depth_frame_mm_key]
            bgr_bgr_frame      = bgr_bgr  [bgr_bgr_frame_key]

            bgr_depth_frame_float_m = bgr_depth_frame_mm.astype(np.float32) / 1000.0

        depth_mask = depth_frame_mm > 0 # if depth_frame_mm is zero, it means a missing value from lack of GT model (e.g. of room walls)
        # these x_flow and y_flow values are filled with NaN
        depth_frame_float_m = depth_frame_mm.astype(np.float32) / 1000.0

        # Initial point cloud
        Z_m = depth_frame_float_m
        X_m = map1 * Z_m
        Y_m = map2 * Z_m
        left_XYZ = np.dstack((X_m, Y_m, Z_m))

        # Get poses of objects and camera in vicon frame
        left_time = frame_info['cam']['ts']

        if dframes is None:
            right_time = left_time + flow_dt
        else:
            right_time_index = i*iterskip + iterskip
            if right_time_index >= len(frames_with_depth):
                right_time_index = len(frames_with_depth) - 1
                if not quiet: print('clipping right_time_index')
            right_time = frames_with_depth[right_time_index]['cam']['ts']
        left_poses = {}
        right_poses = {}
        for key in all_poses:
            left_poses[key]  = interpolate_pose(left_time,  all_poses[key])
            right_poses[key] = interpolate_pose(right_time, all_poses[key])

        # Get the poses of objects in camera frame
        # at the left and right times
        left_to_right_pose = {}
        left_to_bgr_pose = {}
        for key in all_poses.keys():
            if key == 'cam':
                continue

            T_c1o = left_poses[key]
            T_c2o = right_poses[key]

            T_c2c1 = apply_transform(T_c2o, inv_transform(T_c1o))

            left_to_right_pose[key] = T_c2c1

            if bgr_file is not None:
                T_c2o = interpolate_pose(bgr_frame_info['ts'], all_poses[key])
                T_c2c1 = apply_transform(T_c2o, inv_transform(T_c1o))
                T_r2c1 = apply_transform(inv_transform(T_cr), T_c2c1)
                left_to_bgr_pose[key] = T_r2c1

        # Flatten for transformation
        left_XYZ_flat_cols  = left_XYZ.reshape(X_m.shape[0]*X_m.shape[1], 3).transpose()
        right_XYZ_flat_rows = np.zeros(left_XYZ_flat_cols.shape, dtype=np.float32).transpose()

        if bgr_file is not None:
            bgr_right_XYZ_flat_rows = np.zeros(left_XYZ_flat_cols.shape, dtype=np.float32).transpose()

        # Transform the points for each object
        for key in left_to_right_pose.keys():
            R_rl = R.from_quat(left_to_right_pose[key][4:8]).as_matrix().astype(np.float32)
            t_rl = left_to_right_pose[key][1:4].astype(np.float32)

            # If this conversion does not work, there is a problem with the data
            object_id = int(key)
            mask_id = 1000 * object_id

            object_points = (mask_frame == mask_id).flatten()

            p_l = left_XYZ_flat_cols[:, object_points]

            p_r = (R_rl @ p_l).transpose() + t_rl

            right_XYZ_flat_rows[object_points, :] = p_r

            if bgr_file is not None:
                T_r2c1 = left_to_bgr_pose[key]
                R_r2c1 = R.from_quat(T_r2c1[4:8]).as_matrix().astype(np.float32)
                t_r2c1 = T_r2c1[1:4].astype(np.float32)

                p_c1 = left_XYZ_flat_cols[:, object_points]
                p_r2 = (R_r2c1 @ p_c1).transpose() + t_r2c1

                bgr_right_XYZ_flat_rows[object_points, :] = p_r2

        # Reshape into 3 channel image
        right_XYZ = right_XYZ_flat_rows.reshape(X_m.shape[0], X_m.shape[1], 3)

        if bgr_file is not None:
            bgr_XYZ = bgr_right_XYZ_flat_rows.reshape(X_m.shape[0], X_m.shape[1], 3)

        # Project right_XYZ back to the distorted camera frame
        W_x, W_y = project_points_radtan(right_XYZ,
                                         K[0, 0], K[1,1], K[0, 2], K[1, 2],
                                         *dist_coeffs)

        # Calculate flow, mask out points where depth is unknown
        dx = W_x - xx
        dy = W_y - yy
        dx[depth_mask==0]=float("nan")
        dy[depth_mask==0]=float("nan")

        if bgr_file is not None:
            p_x, p_y = project_points_radtan(bgr_XYZ,
                                             bgr_K[0, 0], bgr_K[1, 1], bgr_K[0, 2], bgr_K[1, 2],
                                             *bgr_dist_coeffs)
            bgr_in_c_0 = cv2.remap(bgr_bgr_frame, p_x, p_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)

            # Only reproject if we had depth at the event cameras pixel
            # and the 3D world point is not occluded in the bgr camera's view
            bgr_depth_in_c_0 = cv2.remap(bgr_depth_frame_float_m, p_x.astype(np.float32), p_y.astype(np.float32), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)

            # Use 3 mm threshold since the depth maps are saved with  a resolution of 1mm
            # thus the error could be +/- 2 mm and a slight tolerance is needed on top of that
            # without this there is visible depth aliasing
            valid_reprojections = bgr_XYZ[:, :, 2] < bgr_depth_in_c_0 + 0.003
            bgr_in_c_0_valid = depth_mask * valid_reprojections

            bgr_in_c = np.zeros(bgr_in_c_0.shape, dtype=bgr_in_c_0.dtype)
            bgr_in_c[bgr_in_c_0_valid, :] = bgr_in_c_0[bgr_in_c_0_valid, :]

        # Watch for time gaps in data
        relative_time=left_time
        if not np.isnan(last_time):
            this_delta_time=relative_time-last_time
            delta_times.append(this_delta_time)
            median_delta_time=statistics.median(delta_times)
            if this_delta_time > 2*median_delta_time :
                factor=this_delta_time/median_delta_time
                if not quiet: print(f'Warning: This delta time {this_delta_time:.3f}s at relative time {relative_time:.3f}s is {factor:.1f}X more than the median delta time {median_delta_time:.3f}s')
                large_delta_times+=1
            last_delta_time=this_delta_time
        else:
            start_time=relative_time
        last_time=relative_time

        # Save results
        timestamps[i]        = relative_time + ros_time_offset
        end_timestamps[i]    = right_time    + ros_time_offset
        if format == 'evimo2v1':
            x_flow_dist[i, :, :] = dx
            y_flow_dist[i, :, :] = dy
        elif format == 'evimo2v2':
            flow_dist = np.stack((dx, dy), axis=-1).astype(np.float32) # float32 is more than accurate enough and saves a lot of memory/compression time
            flow_key = 'flow_' + str(frame_info['id'] - first_frame_id).rjust(10, '0')
            add_to_npz(npz_flow_file, flow_key, flow_dist)
            if bgr_file is not None:
                bgr_key      = 'reprojected_classical_'     + str(frame_info['id'] - first_frame_id).rjust(10, '0')
                bgr_mask_key = 'reprojected_classical_mask' + str(frame_info['id'] - first_frame_id).rjust(10, '0')
                add_to_npz(npz_bgr_file, bgr_key,      bgr_in_c)
                add_to_npz(npz_bgr_file, bgr_mask_key, bgr_in_c_0_valid)
        else:
            raise Exception('Unsupported EVIMO format')

        if showflow:
            # Visualize
            cv2.imshow('mask', mask_frame)

            flow_hsv = visualize_optical_flow(np.dstack((dx, dy)))
            flow_bgr = cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2BGR)

            draw_flow_arrows(flow_bgr, xx, yy, dx, dy)
            cv2.imshow('flow_bgr', flow_bgr)

            if bgr_file is not None:
                cv2.imshow('bgr', cv2.resize(bgr_bgr_frame, (int(bgr_bgr_frame.shape[1] / 4), int(bgr_bgr_frame.shape[0] / 4))))
                vis_bgr = (0.8*bgr_in_c + 0.2*flow_bgr).astype(np.uint8)
                vis_bgr[~bgr_in_c_0_valid, :] = [255, 255, 255]
                cv2.imshow('reprojected bgr + flow', vis_bgr)
                cv2.imshow('reprojected bgr', bgr_in_c)

            cv2.waitKey(waitKey)

    del meta, depth, mask
    if not quiet:print(f'Relative start time {start_time:.3f}s, last time {last_time:.3f}s, duration is {(last_time-start_time):.3f}s.\n'
                       f'There are {large_delta_times} ({(100*(large_delta_times/len(timestamps))):.1f}%) excessive delta times.\n')

    if format == 'evimo2v1':
        if not quiet: print(f'Saving {len(timestamps)} frames in NPZ file {npz_name_base}.npz...')
        np.savez_compressed(
            npz_name_base,
            x_flow_dist=x_flow_dist,
            y_flow_dist=y_flow_dist,
            timestamps=timestamps,
            end_timestamps=end_timestamps)
    elif format == 'evimo2v2':
        add_to_npz(npz_flow_file, 't',     timestamps)
        add_to_npz(npz_flow_file, 't_end', end_timestamps)
        close_npz(npz_flow_file)
        if bgr_file is not None:
            add_to_npz(npz_bgr_file, 't', timestamps)
            close_npz(npz_bgr_file)
    else:
        raise Exception('Unsupported EVIMO format')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(epilog='Calculates optical flow from EVIMO datasets. '
                                            'Each of the source npz files on the command line is processed to produce the corresponding flow npz files. '
                                            'See source code for details of output.')
    parser.add_argument('--dt', dest='dt', help='dt for flow approximation'
                      '"dt" is how far ahead of the camera trajectory to sample in seconds'
                    'when approximating flow through finite difference. Smaller values are'
                    ' more accurate, but noiser approximations of optical flow. '
                    'The flow velocity is obtained from dx,dy/dt, where dx,dy are written to the flow output files', type=float,  default=0.01)
    parser.add_argument('--quiet', help='turns off prints from convert function', default=False, action='store_true')
    parser.add_argument('--visualize', help='Visualize and display results in OpenCV window', default=False, action='store_true')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true', help='Overwrite existing output files')
    parser.add_argument('--wait', dest='wait', action='store_true', help='Wait for keypress between visualizations (for debugging)')
    parser.add_argument('--dframes', dest='dframes', type=int, default=None, help='Alternative to flow_dt, flow is calculated for time N depth frames ahead. '
                                                                                  'Useful because the resulting displacement arrows point to the new position of points '
                                                                                  'in the scene at the time of a ground truth frame in the future')
    parser.add_argument('--format', dest='format', type=str, help='"evimo2v1" or "evimo2v2" input data format')
    parser.add_argument('--reprojectbgr', dest='reproject_bgr', action='store_true', help='Reproject a classical camera measurement into the flow frame')
    parser.add_argument('files', nargs='*',help='NPZ files to convert')

    args = parser.parse_args()
    files=args.files

    if len(files)==0:
        import easygui
        files=[easygui.fileopenbox()]
        if files[0] is None:
            print('nothing to convert. Use -h flag for usage.')
            quit(0)


    p_args_list = []
    for f in files:
        # Assume files are in the standard structure
        bgr_file = None
        if args.reproject_bgr:
            for c in ['left_camera', 'right_camera', 'samsung_mono']:
                if c in f: bgr_file = f.replace(c, 'flea3_7')
        p_args_list.append([
            f,
            args.dt,
            args.quiet,
            args.visualize,
            args.overwrite,
            int(not(args.wait)),
            args.dframes,
            args.format,
            bgr_file
        ])

    with Pool(multiprocessing.cpu_count()) as p:
       list(tqdm(p.istarmap(convert, p_args_list), total=len(p_args_list), position=0, desc='Sequences'))
