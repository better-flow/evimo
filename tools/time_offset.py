#!/usr/bin/python3

import argparse
import numpy as np
import os, sys, shutil, signal, glob, time
import matplotlib.colors as colors

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2
from utils import *


def getBF(fin):
    f = open(fin, 'r')
    lines = f.readlines()
    f.close()

    irosT_x = []
    irosT_y = []
    irosT_z = []
    irosQ_z = []

    bf_ts = []
    for line in lines:
        spl = line.split(' ')
        try:
            bf_ts.append(float(spl[0]))
            irosT_y.append(-float(spl[1]))
            irosT_x.append(-float(spl[2]))
            irosT_z.append(-float(spl[3]))
            irosQ_z.append(float(spl[4]))
        except:
            continue
    
    return np.array(irosT_x), np.array(irosT_y), np.array(irosT_z), np.array(irosQ_z), np.array(bf_ts)


def clean_outliers(array, th):
    for i in range(1, array.shape[0]):
        if (abs(array[i] - array[i - 1]) > th):
            array[i] = array[i - 1]


def scale_factor(a1, a2, th):
    a1_abs = np.abs(a1)
    a2_abs = np.abs(a2)

    avg = 0.0
    cnt = 0.0
    for i in range(0, min(a2_abs.shape[0], a1_abs.shape[0])):
        if (a2_abs[i] < th):
            continue
        avg += a1_abs[i] / a2_abs[i]
        cnt += 1.0

    if (cnt == 0):
        avg = 1.0
        ant = 1.0
    return avg, cnt




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        type=str,
                        default='.',
                        required=False)
    parser.add_argument('--width',
                        type=float,
                        required=False,
                        default=0.05)
    parser.add_argument('--offset',
                        type=float,
                        required=False,
                        default=0.0)
    parser.add_argument('--mode',
                        type=int,
                        required=False,
                        default=0)

    args = parser.parse_args()

    print ("Opening", args.base_dir)

  
    # Trajectories
    cam_traj_global = read_camera_traj(os.path.join(args.base_dir, 'trajectory.txt'))

    sl_npz = np.load(args.base_dir + '/recording.npz')
    #cloud          = sl_npz['events']
    #idx            = sl_npz['index']
    #discretization = sl_npz['discretization']
    #K              = sl_npz['K']
    #D              = sl_npz['D']
    #depth_gt       = sl_npz['depth']
    #mask_gt        = sl_npz['mask']
    gt_ts          = sl_npz['gt_ts']

    if (len(cam_traj_global.keys()) != len(gt_ts)):
        print("Camera vs Timestamp counts differ!")
        print("\t", len(cam_traj_global.keys()), len(gt_ts))

    cam_vels = cam_poses_to_vels(cam_traj_global, gt_ts)

    # plot
    import matplotlib.pyplot as plt

    gT_z = np.array([cam_vels[i][0][0] for i in sorted(cam_vels.keys())])
    gT_x = np.array([cam_vels[i][0][1] for i in sorted(cam_vels.keys())])
    gT_y = np.array([cam_vels[i][0][2] for i in sorted(cam_vels.keys())])
    gEuler = [quaternion_to_euler(cam_vels[i][1]) for i in sorted(cam_vels.keys())]
    gQ_x = np.array([q[0] for q in gEuler])
    gQ_z = np.array([q[1] for q in gEuler])
    gQ_y = np.array([q[2] for q in gEuler])


    bfT_x, bfT_y, bfT_z, bfQ_z, bf_ts = getBF(os.path.join(args.base_dir, 'bf_egomotion.txt'))
    bfQ_x = np.zeros(bfQ_z.shape)
    bfQ_y = np.zeros(bfQ_z.shape)

    th = np.max(np.sqrt(gT_x * gT_x)) * 0.9
    s1, c1 = scale_factor(np.sqrt(bfT_x * bfT_x), np.sqrt(gT_x * gT_x), th)
    bfT_x *= (c1) / (s1)

    th = np.max(np.sqrt(gT_y * gT_y)) * 0.9
    s1, c1 = scale_factor(np.sqrt(bfT_y * bfT_y), np.sqrt(gT_y * gT_y), th)
    bfT_y *= (c1) / (s1)

    th = np.max(np.sqrt(gT_z * gT_z)) * 0.9
    s1, c1 = scale_factor(np.sqrt(bfT_z * bfT_z), np.sqrt(gT_z * gT_z), th)
    bfT_z *= (c1) / (s1)


    fig, axs = plt.subplots(6, 1)

    # In standard camera frame:
    # Translation
    # X axis
    axs[0].plot(gt_ts,  gT_x)
    axs[0].plot(bf_ts, bfT_x)
    
    # Y axis
    axs[1].plot(gt_ts,  gT_y)
    axs[1].plot(bf_ts, bfT_y)
    axs[1].set_ylabel('camera linear (m/s)')

    # Z axis
    axs[2].plot(gt_ts,  gT_z)
    axs[2].plot(bf_ts, bfT_z)
    
    # Rotation
    # X axis
    axs[3].plot(gt_ts,  gQ_x)
    axs[3].plot(bf_ts, bfQ_x)
    
    # Y axis
    axs[4].plot(gt_ts,  gQ_y)
    axs[4].plot(bf_ts, bfQ_y)
    axs[4].set_ylabel('camera angular (deg/s)')

    # Z axis
    axs[5].plot(gt_ts,  gQ_z)
    axs[5].plot(bf_ts, bfQ_z)
    axs[5].set_xlabel('time, s')


    plt.savefig(os.path.join(args.base_dir, 'time_alignment_plots.svg'))
    plt.show()
