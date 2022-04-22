#!/usr/bin/python3
import argparse
import cv2
import numpy as np
import glob
import pickle
import os
import argparse
import math

from extract_event_and_frame_times import (get_sequence_name,
                                           get_camera_name,
                                           get_category,
                                           get_purpose)

def group_files_by_sequence_name(files):
    files_grouped = {}

    for file in files:
        sequence_name = get_sequence_name(file)
        camera_name = get_camera_name(file)
        category = get_category(file)
        purpose = get_purpose(file)

        if sequence_name in files_grouped:
            files_grouped[sequence_name][3][camera_name] = file
        else:
            files_grouped[sequence_name] = [category, purpose, sequence_name, {camera_name: file}]

    return files_grouped

def calc_fov(calib):
    print(calib)

    cx = calib['cx']
    cy = calib['cy']
    fx = calib['fx']
    fy = calib['fy']

    res_x = calib['res_x']
    res_y = calib['res_y']


    K = np.array(((fx,  0, cx),
                  ( 0, fy, cy),
                  ( 0,  0,  1)))
    K_inv = np.linalg.inv(K)

    p1 = np.array((0, 0, 1))
    p2 = np.array((res_x, res_y, 1))

    X1_over_Z1 = K_inv @ p1
    X2_over_Z2 = K_inv @ p2

    X1_hat = X1_over_Z1 / np.linalg.norm(X1_over_Z1)
    X2_hat = X2_over_Z2 / np.linalg.norm(X2_over_Z2)

    theta = math.acos((X1_hat @ X2_hat))

    theta_deg = theta * 180 / math.pi
    print(theta_deg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View all cameras of a sequence with GT depth overlaid to see availability')
    parser = argparse.ArgumentParser()
    parser.add_argument('--idir', type=str, help='Directory containing npz file tree')
    parser.add_argument('--seq', default='scene13_dyn_test_00', help='Sequence name')
    args = parser.parse_args()

    data_base_folder = args.idir
    file_glob = data_base_folder + '/*/*/*/*'
    files = sorted(list(glob.glob(file_glob)))

    files_grouped_by_sequence = group_files_by_sequence_name(files)
    folders = files_grouped_by_sequence[args.seq][3]

    print('Opening npy files')
    if 'flea3_7' in folders:
        print('flea3_7_dataset_info')
        meta = np.load(os.path.join(folders['flea3_7'], 'dataset_info.npz'), allow_pickle=True)['meta'].item()
        calibration = meta['meta']
        calc_fov(calibration)

    if 'left_camera' in folders:
        print('left_camera dataset_info')
        meta = np.load(os.path.join(folders['left_camera'], 'dataset_info.npz'), allow_pickle=True)['meta'].item()
        calibration = meta['meta']
        calc_fov(calibration)

    if 'right_camera' in folders:
        print('right_camera dataset_info')
        meta = np.load(os.path.join(folders['right_camera'], 'dataset_info.npz'), allow_pickle=True)['meta'].item()
        calibration = meta['meta']
        calc_fov(calibration)

    if 'samsung_mono' in folders:
        print('samsung_mono dataset_info')
        meta = np.load(os.path.join(folders['samsung_mono'], 'dataset_info.npz'), allow_pickle=True)['meta'].item()
        calibration = meta['meta']
        calc_fov(calibration)

    flea3_7_resolution = (1552, 2080)
    left_camera_resolution  = (480, 640)
    right_camera_resolution = (480, 640)
    samsung_mono_resolution = (480, 640)
