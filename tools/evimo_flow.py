###############################################################################
#
# File: evimo_flow_converter.py
#
# Calculate normal flow on EVIMO datasets
# so far this is tested on EVIMO2
#
# Usage:
# python evimo_flow.py --file evimo_file.npz --dt 0.01
#
# Writes out flow files to:
# evimo_file_flow.npz
# which contains:
# timestamps.npy
# x_flow.npy
# y_flow.npy
#
# "dt" is how far ahead of the camera trajectory to sample in seconds
# when approximating flow through finite difference. Smaller values are
# more accurate, but noiser approximations of optical flow.
#
# History:
# 01-21-22 - Levi Burner - Created file
#
###############################################################################

import argparse
import cv2
import numpy as np
import pprint
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# Sample a list of translations and rotations
# with linear interpolation
def interpolate_cam_pose(t, cam_pose):
    right_i = np.searchsorted(cam_pose[:, 0], t)

    left_t  = cam_pose[right_i-1, 0]
    right_t = cam_pose[right_i,   0]

    alpha = (t - left_t) / (right_t - left_t)

    left_position  = cam_pose[right_i - 1, 1:4]
    right_position = cam_pose[right_i,     1:4]

    position_interp = alpha * (right_position - left_position) + left_position

    left_right_rot_stack = R.from_quat((cam_pose[right_i - 1, 4:8],
                                        cam_pose[right_i,     4:8]))

    slerp = Slerp((0, 1), left_right_rot_stack)
    R_interp = slerp(alpha)

    return np.array([t,] + list(position_interp) + list(R_interp.as_quat()))

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
    y__ = y_ * dist + 2.0 * p2 * x_ * y_ + p1 * (r2 * 2.0 * y_ * y_)


    u = fx * x__ + cx
    v = fy * y__ + cy

    return u, v

# Generate an HSV image using color to represent the gradient direction in a optical flow field
def visualize_optical_flow(flow):
    theta = np.mod(np.arctan2(flow[:, :, 0], flow[:, :, 1]) + 2*np.pi, 2*np.pi)

    flow_norms = np.linalg.norm(flow, axis=2)
    flow_norms_normalized = flow_norms / np.max(flow_norms)

    flow_hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flow_hsv[:, :, 0] = 179 * theta / (2*np.pi)
    flow_hsv[:, :, 1] = 255 * flow_norms_normalized
    flow_hsv[:, :, 2] = 255 * (flow_norms_normalized > 0)

    return flow_hsv

def draw_flow_arrows(img, xx, yy, flow_x, flow_y, p_skip=10):
    xx     = xx[::p_skip, ::p_skip].flatten()
    yy     = yy[::p_skip, ::p_skip].flatten()
    flow_x = dx[::p_skip, ::p_skip].flatten()
    flow_y = dy[::p_skip, ::p_skip].flatten()

    for x, y, u, v in zip(xx, yy, flow_x, flow_y):
        cv2.arrowedLine(img,
                        (int(x), int(y)),
                        (int(x+u), int(y+v)),
                        (255, 255, 255),
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

# Get a list of camera translations and rotations in the vicon frame
# out of EVIMO's 'meta' field
def get_cam_poses(meta):
    cam_pose_samples = len(meta['full_trajectory'])
    cam_poses = np.zeros((cam_pose_samples, 1+3+4))

    # Convert camera poses to array
    for cam_pose, all_pose in zip(cam_poses, meta['full_trajectory']):
        assert 'cam' in all_pose

        cam_pose[0] = all_pose['cam']['ts']
        cam_pose[1] = all_pose['cam']['pos']['t']['x']
        cam_pose[2] = all_pose['cam']['pos']['t']['y']
        cam_pose[3] = all_pose['cam']['pos']['t']['z']

        cam_pose[4] = all_pose['cam']['pos']['q']['x']
        cam_pose[5] = all_pose['cam']['pos']['q']['y']
        cam_pose[6] = all_pose['cam']['pos']['q']['z']
        cam_pose[7] = all_pose['cam']['pos']['q']['w']

    return cam_poses

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', dest='file', help='EVIMO2 npz file to process',
                         default='left_camera/imo/train/scene13_dyn_test_01.npz')
    parser.add_argument('--dt', dest='dt', help='dt for flow approximation', type=float,
                        default=0.01)
    args = parser.parse_args()

    file    = args.file
    flow_dt = args.dt

    data = np.load(file, allow_pickle=True)
    meta = data['meta'].item()

    cam_poses      = get_cam_poses (meta)
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

    # For visualization
    flow_direction_image_hsv = flow_direction_image((data['depth'][0].shape[0], data['depth'][0].shape[1]))
    flow_direction_image_bgr = cv2.cvtColor(flow_direction_image_hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('color direction chart', flow_direction_image_bgr)

    # Preallocate arrays for flow as in MSEVC format
    timestamps  = np.zeros((data['depth'].shape[0],), dtype=np.float32)
    x_flow = np.zeros((data['depth'].shape), dtype=np.float32)
    y_flow = np.zeros((data['depth'].shape), dtype=np.float32)

    for i, (depth_frame_mm, frame_info) in enumerate(zip(data['depth'], meta['frames'])):
        depth_mask = depth_frame_mm > 0
        depth_frame_float_m = depth_frame_mm.astype('float') / 1000.0

        # Initial point cloud
        Z_m = depth_frame_float_m
        X_m = map1 * Z_m
        Y_m = map2 * Z_m
        left_XYZ = np.dstack((X_m, Y_m, Z_m))

        # Get the change in coordinate frame
        left_time = frame_info['cam']['ts']
        right_time = left_time + flow_dt

        left_pose  = interpolate_cam_pose(left_time,  cam_poses)
        right_pose = interpolate_cam_pose(right_time, cam_poses)

        delta_t = left_pose[1:4] - right_pose[1:4]

        R_left  = R.from_quat(left_pose [4:8])
        R_right = R.from_quat(right_pose[4:8])
        right_R_inv = R_right.inv().as_matrix()
        delta_R = (R_right.inv() * R_left).as_matrix()

        # Flatten for transformation
        left_XYZ_flat_cols  = left_XYZ.reshape(X_m.shape[0]*X_m.shape[1], 3).transpose()

        # Transform the point cloud
        right_XYZ_flat_rows = (delta_R @ left_XYZ_flat_cols).transpose() + right_R_inv @ delta_t

        # Reshape into 3 channel image
        right_XYZ           = right_XYZ_flat_rows.reshape(X_m.shape[0], X_m.shape[1], 3)

        # Project right_XYZ back to the distorted camera frame
        W_x, W_y = project_points_radtan(right_XYZ,
                                         K[0, 0], K[1,1], K[0, 2], K[1, 2],
                                         *dist_coeffs)

        # Calculate flow, mask out points where depth is unknown
        dx = (W_x - xx) * depth_mask
        dy = (W_y - yy) * depth_mask

        timestamps[i] = left_time
        x_flow[i, :, :] = dx
        y_flow[i, :, :] = dy

        # Visualize
        flow_hsv = visualize_optical_flow(np.dstack((dx, dy)))
        flow_bgr = cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2BGR)

        draw_flow_arrows(flow_bgr, xx, yy, dx, dy)
        cv2.imshow('flow_bgr', flow_bgr)

        cv2.waitKey(1)

    np.savez_compressed(
            file[:-4] + '_flow',
            x_flow=x_flow,
            y_flow=y_flow,
            timestamps=timestamps)
