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
    obj_traj_global = read_object_traj(os.path.join(args.base_dir, 'objects.txt'))
    
    nums = sorted(obj_traj_global.keys())
    oids = []
    if (len(nums) > 0):
        oids = sorted(obj_traj_global[nums[0]].keys())
    
    nums = sorted(cam_traj_global.keys())
    if (len(oids) > 0 and len(cam_traj_global.keys()) != len(obj_traj_global.keys())):
        print("Camera vs Obj pose numbers differ!")
        print("\t", len(cam_traj_global.keys()), len(obj_traj_global.keys()))
        sys.exit()

    obj_traj = to_cam_frame(obj_traj_global, cam_traj_global);

    sl_npz = np.load(args.base_dir + '/recording.npz')
    cloud          = sl_npz['events']
    idx            = sl_npz['index']
    discretization = sl_npz['discretization']
    K              = sl_npz['K']
    D              = sl_npz['D']
    depth_gt       = sl_npz['depth']
    mask_gt        = sl_npz['mask']
    gt_ts          = sl_npz['gt_ts']

    if (len(cam_traj_global.keys()) != len(gt_ts)):
        print("Camera vs Timestamp counts differ!")
        print("\t", len(cam_traj_global.keys()), len(gt_ts))

    obj_vels = obj_poses_to_vels(obj_traj, gt_ts)
    cam_vels = cam_poses_to_vels(cam_traj_global, gt_ts)

    # plot stuff
    import matplotlib.pyplot as plt

    T_x = [cam_vels[i][0][0] for i in sorted(cam_vels.keys())]
    T_y = [cam_vels[i][0][1] for i in sorted(cam_vels.keys())]
    T_z = [cam_vels[i][0][2] for i in sorted(cam_vels.keys())]
    Euler = [quaternion_to_euler(cam_vels[i][1]) for i in sorted(cam_vels.keys())]
    Q_x = [q[0] for q in Euler]
    Q_y = [q[1] for q in Euler]
    Q_z = [q[2] for q in Euler]

    fig, axs = plt.subplots(2 * (len(oids) + 1), 1)
    axs[0].plot(sorted(cam_vels.keys()), T_x)
    axs[0].plot(sorted(cam_vels.keys()), T_y)
    axs[0].plot(sorted(cam_vels.keys()), T_z)
    axs[0].set_xlabel('frame')
    axs[0].set_ylabel('camera linear (m/s)')
    axs[1].plot(sorted(cam_vels.keys()), Q_x)
    axs[1].plot(sorted(cam_vels.keys()), Q_y)
    axs[1].plot(sorted(cam_vels.keys()), Q_z)
    axs[1].set_xlabel('frame')
    axs[1].set_ylabel('camera angular (deg/s)')

    x_axis = [i for i in sorted(obj_vels.keys())]

    for k, id_ in enumerate(oids):
        vels = [obj_vels[i][id_] for i in x_axis]
        
        T_x = np.array([vel[0][0] for vel in vels])
        T_y = np.array([vel[0][1] for vel in vels])
        T_z = np.array([vel[0][2] for vel in vels])
        Euler = [quaternion_to_euler(vel[1]) for vel in vels]
        Q_x = [q[0] for q in Euler]
        Q_y = [q[1] for q in Euler]
        Q_z = [q[2] for q in Euler]

        axs[2 * k + 2].plot(sorted(cam_vels.keys()), T_x)
        axs[2 * k + 2].plot(sorted(cam_vels.keys()), T_y)
        axs[2 * k + 2].plot(sorted(cam_vels.keys()), T_z)
        axs[2 * k + 2].set_xlabel('frame')
        axs[2 * k + 2].set_ylabel('object_' + str(id_) + ' linear (m/s)')
        axs[2 * k + 3].plot(sorted(cam_vels.keys()), Q_x)
        axs[2 * k + 3].plot(sorted(cam_vels.keys()), Q_y)
        axs[2 * k + 3].plot(sorted(cam_vels.keys()), Q_z)
        axs[2 * k + 3].set_xlabel('frame')
        axs[2 * k + 3].set_ylabel('object_' + str(id_) + ' angular (deg/s)')

        f = open(os.path.join(args.base_dir, 'o' + str(id_) + '_filter.txt'), 'w')
        for i in range(len(T_x)):
            if (T_x[i] * T_x[i] + T_y[i] * T_y[i] + T_z[i] * T_z[i] > 0.05):
                continue
            f.write(str(i) + " " + str(T_x[i]) + " " + str(T_y[i]) + " " + str(T_z[i])
                    + " " + str(Q_x[i]) + " " + str(Q_y[i]) + " " + str(Q_z[i]) + '\n'
            )
        f.close()

    plt.savefig(os.path.join(args.base_dir, 'velocity_plots.svg'))
    #plt.show()
