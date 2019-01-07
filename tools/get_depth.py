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


def gen_text_stub(shape_y, cam_vel, objs_pos, objs_vel):
    oids = []
    if (objs_pos is not None):
        oids = objs_pos.keys()
    step = 20
    shape_x = (len(oids) + 2) * step + 10
    cmb = np.zeros((shape_x, shape_y, 3), dtype=np.float32)

    i = 0
    text = "Camera velocity = " + str(cam_vel[0])
    cv2.putText(cmb, text, (10, 20 + i * step), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    
    for id_ in oids:
        i += 1
        text = str(id_) + ": T = " + str(objs_pos[id_][0]) + " | V = " + str(objs_vel[id_][0])        
        cv2.putText(cmb, text, (10, 20 + i * step), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    return cmb


def dvs_img(cloud, shape, K, D, slice_width):
    cmb = pydvs.dvs_img(cloud, shape, K=K, D=D)

    cmb[:,:,0] *= 50
    cmb[:,:,1] *= 255.0 / slice_width
    cmb[:,:,2] *= 50

    return cmb


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
        #sys.exit()

    obj_vels = obj_poses_to_vels(obj_traj, gt_ts)
    cam_vels = cam_poses_to_vels(cam_traj_global, gt_ts)

    slice_width = args.width

    first_ts = cloud[0][0]
    last_ts = cloud[-1][0]

    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    print (fx, fy, cx, cy) 

    print ("K and D:")
    print (K)
    print (D)
    print ("")

    slice_dir = os.path.join(args.base_dir, 'slices')
    vis_dir   = os.path.join(args.base_dir, 'vis')

    clear_dir(slice_dir)
    clear_dir(vis_dir)

    print ("The recording range:", first_ts, "-", last_ts)
    print ("The gt range:", gt_ts[0], "-", gt_ts[-1])
    print ("Discretization resolution:", discretization)
 
    if (gt_ts[0] > 1.0):
        print("Time offset between events and image frames is too big:", gt_ts[0], "s.")
        gt_ts[:] -= gt_ts[0]


    for i, time in enumerate(gt_ts):
        if (time > last_ts or time < first_ts):
            continue

        depth = depth_gt[i]
        mask  = mask_gt[i]

        sl, _ = pydvs.get_slice(cloud, idx, time, args.width, args.mode, discretization)

        eimg = dvs_img(sl, global_shape, K, D, args.width)
        cimg = eimg[:,:,0] + eimg[:,:,2]

        cv2.imwrite(os.path.join(slice_dir, 'frame_' + str(i).rjust(10, '0') + '.png'), eimg)
        cv2.imwrite(os.path.join(slice_dir, 'depth_' + str(i).rjust(10, '0') + '.png'), depth.astype(np.uint16))
        cv2.imwrite(os.path.join(slice_dir, 'mask_'  + str(i).rjust(10, '0') + '.png'), mask.astype(np.uint16))

        nmin = np.nanmin(depth)
        nmax = np.nanmax(depth)

        #print (np.nanmin(depth), np.nanmax(depth))
        eimg[:,:,2] = (depth - nmin) / (nmax - nmin) * 255

        col_mask = mask_to_color(mask)
        col_mask += np.dstack((cimg, cimg * 0, cimg * 0)) * 1.0

        eimg = np.hstack((eimg, col_mask))
        
        if (len(oids) > 0):
            col_vel = vel_to_color(mask, obj_vels[nums[i]])
            eimg = np.hstack((eimg, col_vel))

            footer = gen_text_stub(eimg.shape[1], cam_vels[nums[i]], obj_traj[nums[i]], obj_vels[nums[i]])
            eimg = np.vstack((eimg, footer))

        if (len(nums) > i):
            footer = gen_text_stub(eimg.shape[1], cam_vels[nums[i]], None, None)
            eimg = np.vstack((eimg, footer))

        cv2.imwrite(os.path.join(vis_dir, 'frame_' + str(i).rjust(10, '0') + '.png'), eimg)
