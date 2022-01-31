#!/usr/bin/python
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
#
###############################################################################

import argparse
import cv2
import numpy as np
import pprint
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import easygui
from tqdm import tqdm
from pathlib import Path
import statistics

# Sample a list of translations and rotations
# with linear interpolation
def interpolate_pose(t, pose):
    right_i = np.searchsorted(pose[:, 0], t)
    if right_i==pose.shape[0]:
        print(f'attempted extrapolation past end of cam poses, clipping')
        right_i=right_i-1 # prevent attempted extrapolation past array end

    left_t  = pose[right_i-1, 0]
    right_t = pose[right_i,   0]

    alpha = (t - left_t) / (right_t - left_t)
    if alpha>1:
        print('attempted alpha>1, clipping')
        alpha=1

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
    x_ = points[:, :, 0] / points[:, :, 2]
    y_ = points[:, :, 1] / points[:, :, 2]
    
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
    for key in meta['full_trajectory'][0].keys():
        if key == 'id' or key == 'ts' or key == 'gt_frame':
            continue
        poses[key] = np.zeros((vicon_pose_samples, 1+3+4))

    # Convert camera poses to array
    for i, all_pose in enumerate(meta['full_trajectory']):
        for key in poses.keys():
            if key == 'id' or key == 'ts' or key == 'gt_frame':
                continue

            assert key in all_pose

            poses[key][i, 0] = all_pose['ts']
            poses[key][i, 1] = all_pose[key]['pos']['t']['x']
            poses[key][i, 2] = all_pose[key]['pos']['t']['y']
            poses[key][i, 3] = all_pose[key]['pos']['t']['z']
            poses[key][i, 4] = all_pose[key]['pos']['q']['x']
            poses[key][i, 5] = all_pose[key]['pos']['q']['y']
            poses[key][i, 6] = all_pose[key]['pos']['q']['z']
            poses[key][i, 7] = all_pose[key]['pos']['q']['w']

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


def convert(file, flow_dt, showflow=True, overwrite=False,
            waitKey=1, dframes=None):
    if file.endswith('_flow.npz'):
        print(f'skipping {file} because it appears to be a flow output npz file')
        return

    npz_name_base=file[:-4] + '_flow'
    p=Path(npz_name_base+'.npz')
    if not overwrite and p.exists():
        print(f'skipping {file} because {p} exists; use --overwrite option to overwrite existing output file')
        return

    if dframes is None:
        print(f'converting {file} with dt={flow_dt}s; loading source...',end='')
    else:
        print(f'converting {file} with dframes={dframes}; loading source...',end='')

    data = np.load(file, allow_pickle=True, mmap_mode='r')
    print('done loading')
    meta = data['meta'].item()

    all_poses      = get_all_poses (meta)
    K, dist_coeffs = get_intrinsics(meta)

    # Get the map from pixels to direction vectors with Z = 1
    map1, map2 = cv2.initInverseRectificationMap(
        K, # Intrinsics
        dist_coeffs, # Distortion
        np.eye(3), # Rectification
        np.eye(3), # New intrinsics
        (data['depth'][0].shape[1], data['depth'][0].shape[0]),
        cv2.CV_32FC1)

    # Initial positions of every point
    x = np.arange(0, data['depth'][0].shape[1], 1)
    y = np.arange(0, data['depth'][0].shape[0], 1)
    xx, yy = np.meshgrid(x, y)

    if showflow:
        # For visualization
        flow_direction_image_hsv = flow_direction_image((data['depth'][0].shape[0], data['depth'][0].shape[1]))
        flow_direction_image_bgr = cv2.cvtColor(flow_direction_image_hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('color direction chart', flow_direction_image_bgr)

    # Preallocate arrays for flow as in MSEVC format
    timestamps      = np.zeros((data['depth'].shape[0],), dtype=np.float64)
    end_timestamps  = np.zeros((data['depth'].shape[0],), dtype=np.float64)
    x_flow_dist = np.zeros((data['depth'].shape), dtype=np.float64) # named as in MVSEC monoocular camera flow, double as in MVSEC NPZs
    y_flow_dist = np.zeros((data['depth'].shape), dtype=np.float64)

    ros_time_offset=meta['meta']['ros_time_offset'] # get offset of start of data in epoch seconds to write the flow timestamps using same timebase as MVSEC
    last_time=float('nan')
    last_delta_time=float('nan')
    delta_times=[]
    start_time=float('nan')
    large_delta_times=0

    if dframes is not None:
        iterskip = dframes
    else:
        iterskip = 1

    for i, (depth_frame_mm, mask_frame, frame_info) in (
            enumerate(tqdm(zip(data['depth'] [::iterskip],
                               data['mask']  [::iterskip],
                               meta['frames'][::iterskip]),
                            total=len(data['depth'][::iterskip])))):
        depth_mask = depth_frame_mm > 0 # if depth_frame_mm is zero, it means a missing value from lack of GT model (e.g. of room walls)
        # these x_flow and y_flow values are filled with NaN
        depth_frame_float_m = depth_frame_mm.astype('float') / 1000.0

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
            right_time_index = iterskip*i+iterskip
            if right_time_index >= len(meta['frames']):
                right_time_index = len(meta['frames']) - 1
                print('clipping right_time_index')
            right_time = meta['frames'][right_time_index]['cam']['ts']

        left_poses = {}
        right_poses = {}
        for key in all_poses:
            left_poses[key]  = interpolate_pose(left_time,  all_poses[key])
            right_poses[key] = interpolate_pose(right_time, all_poses[key])

        # Get the poses of objects in camera frame
        # at the left and right times
        left_to_right_pose = {}
        for key in all_poses.keys():
            if key == 'cam':
                continue

            T_c1o1 = left_poses[key]
            T_c2o2 = right_poses[key]

            T_c2c1 = apply_transform(T_c2o2, inv_transform(T_c1o1))

            left_to_right_pose[key] = T_c2c1

        # Flatten for transformation
        left_XYZ_flat_cols  = left_XYZ.reshape(X_m.shape[0]*X_m.shape[1], 3).transpose()
        right_XYZ_flat_rows = np.zeros(left_XYZ_flat_cols.shape).transpose()

        # Transform the points for each object
        for key in left_to_right_pose.keys():
            R_rl = R.from_quat(left_to_right_pose[key][4:8]).as_matrix()
            t_rl = left_to_right_pose[key][1:4]

            # If this conversion does not work, there is a problem with the data
            object_id = int(key)
            mask_id = 1000 * object_id

            object_points = (mask_frame == mask_id).flatten()

            p_l = left_XYZ_flat_cols[:, object_points]

            p_r = (R_rl @ p_l).transpose() + t_rl

            right_XYZ_flat_rows[object_points, :] = p_r

        # Reshape into 3 channel image
        right_XYZ = right_XYZ_flat_rows.reshape(X_m.shape[0], X_m.shape[1], 3)

        # Project right_XYZ back to the distorted camera frame
        W_x, W_y = project_points_radtan(right_XYZ,
                                         K[0, 0], K[1,1], K[0, 2], K[1, 2],
                                         *dist_coeffs)

        # Calculate flow, mask out points where depth is unknown
        dx = W_x - xx
        dy = W_y - yy
        dx[depth_mask==0]=float("nan")
        dy[depth_mask==0]=float("nan")

        relative_time=left_time
        if not np.isnan(last_time):
            this_delta_time=relative_time-last_time
            delta_times.append(this_delta_time)
            median_delta_time=statistics.median(delta_times)
            if this_delta_time > 2*median_delta_time :
                factor=this_delta_time/median_delta_time
                print(f'Warning: This delta time {this_delta_time:.3f}s at relative time {relative_time:.3f}s is {factor:.1f}X more than the median delta time {median_delta_time:.3f}s')
                large_delta_times+=1
            last_delta_time=this_delta_time
        else:
            start_time=relative_time
        last_time=relative_time

        timestamps[i]     = relative_time + ros_time_offset
        end_timestamps[i] = right_time    + ros_time_offset
        x_flow_dist[i, :, :] = dx
        y_flow_dist[i, :, :] = dy

        if showflow:
            # Visualize
            cv2.imshow('mask', mask_frame)

            flow_hsv = visualize_optical_flow(np.dstack((dx, dy)))
            flow_bgr = cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2BGR)

            draw_flow_arrows(flow_bgr, xx, yy, dx, dy)
            cv2.imshow('flow_bgr', flow_bgr)

            cv2.waitKey(waitKey)
    del data
    print(f'Relative start time {start_time:.3f}s, last time {last_time:.3f}s, duration is {(last_time-start_time):.3f}s.\n'
          f'There are {large_delta_times} ({(100*(large_delta_times/len(timestamps))):.1f}%) excessive delta times.\n'
          f'Saving {len(timestamps)} frames in NPZ file {npz_name_base}.npz...')
    np.savez_compressed(
        npz_name_base,
        x_flow_dist=x_flow_dist,
        y_flow_dist=y_flow_dist,
        timestamps=timestamps,
        end_timestamps=end_timestamps)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(epilog='Calculates optical flow from EVIMO datasets. '
                                            'Each of the source npz files on the command line is processed to produce the corresponding flow npz files. '
                                            'See source code for details of output.')
    parser.add_argument('--dt', dest='dt', help='dt for flow approximation'
                      '"dt" is how far ahead of the camera trajectory to sample in seconds'
                    'when approximating flow through finite difference. Smaller values are'
                    ' more accurate, but noiser approximations of optical flow. '
                    'The flow velocity is obtained from dx,dy/dt, where dx,dy are written to the flow output files', type=float,  default=0.01)
    parser.add_argument('--quiet', help='turns off OpenCV graphical output windows', default=False, action='store_true')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true', help='Overwrite existing output files')
    parser.add_argument('--wait', dest='wait', action='store_true', help='Wait for keypress between visualizations (for debugging)')
    parser.add_argument('--dframes', dest='dframes', type=int, default=None, help='Alternative to flow_dt, flow is calculated for time N depth frames ahead. '
                                                                                  'Useful because the resulting displacement arrows point to the new position of points '
                                                                                  'in the scene at the time of a ground truth frame in the future')
    parser.add_argument('files', nargs='*',help='NPZ files to convert')

    args = parser.parse_args()
    files=args.files

    if len(files)==0:
        import easygui
        files=[easygui.fileopenbox()]


    for f in files:
        convert(f,
                flow_dt=args.dt,
                showflow=not args.quiet,
                overwrite=args.overwrite,
                waitKey=int(not args.wait),
                dframes=args.dframes)
