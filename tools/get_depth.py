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


def vec_to_str(v, q=None):
    ret = ""
    for elem in v:
        if (elem >= 0):
            ret += " "
        ret += "{0:.2f}".format(elem) + " "
    
    if (q is None):
        return ret

    ret += "/ "
    X, Y, Z = quaternion_to_euler(q)

    ret += "{0:.2f}".format(X) + " "
    ret += "{0:.2f}".format(Y) + " "
    ret += "{0:.2f}".format(Z)

    return ret

def vel2text(vel):
    ret = ""
    ret += str(vel[0][0]) + " " 
    ret += str(vel[0][1]) + " " 
    ret += str(vel[0][2]) + " " 

    X, Y, Z = quaternion_to_euler(vel[1])

    ret += str(X) + " " 
    ret += str(Y) + " " 
    ret += str(Z)

    return ret


def gen_text_stub(shape_y, cam_vel, objs_pos, objs_vel):
    oids = []
    if (objs_pos is not None):
        oids = objs_pos.keys()
    step = 20
    shape_x = (len(oids) + 2) * step + 10
    cmb = np.zeros((shape_x, shape_y, 3), dtype=np.float32)

    i = 0
    text = "Camera velocity = " + vec_to_str(cam_vel[0], cam_vel[1])
    cv2.putText(cmb, text, (10, 20 + i * step), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    
    for id_ in oids:
        i += 1
        text = str(id_) + ": T = " + vec_to_str(objs_pos[id_][0]) + " | V = " + vec_to_str(objs_vel[id_][0])        
        cv2.putText(cmb, text, (10, 20 + i * step), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    return cmb


def dvs_img(cloud, shape, K, D, slice_width):
    cmb = pydvs.dvs_img(cloud, shape, K=K, D=D)

    time = cmb[:,:,1]
    pcnt = cmb[:,:,2]
    ncnt = cmb[:,:,0]
    cnt = pcnt + ncnt
    
    #ncnt[cnt < 1.5] = 0
    #pcnt[cnt < 1.5] = 0
    #time[cnt < 1.5] = 0

    cmb[:,:,0] *= 50
    cmb[:,:,1] *= 255.0 / slice_width
    cmb[:,:,2] *= 50

    #tmp = np.copy(cmb)
    #tmp[:,:,0] = cmb[:,:,1]
    #tmp[:,:,1] = cmb[:,:,0]
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

    #print(quaternion_to_euler([ 0.9975021, 0.0499167, -0.0024979, 0.0499167])) 
    #exit()

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


    rgb_dir = os.path.join(args.base_dir, 'img')
    add_rgb = True
    if not os.path.exists(rgb_dir):
        add_rgb = False

    rgb_name_list = []
    if (add_rgb):
        #flist = sorted(os.listdir(rgb_dir))
        flist = np.loadtxt(os.path.join(args.base_dir, 'images.txt'), usecols=1, dtype='str')
        rgb_ts = np.loadtxt(os.path.join(args.base_dir, 'images.txt'), usecols=0)
        print ("Image files:", len(flist), "Image timestamps:", rgb_ts.shape, "Gt ts:", len(gt_ts))

        for i, ts in enumerate(gt_ts):
            nearest_delta = 1000.0
            nearest_idx = -1
            for j, ts_ in enumerate(rgb_ts):
                if (abs(ts - ts_) < nearest_delta):
                    nearest_delta = abs(ts - ts_)
                    nearest_idx = j
            
            rgb_name_list.append(flist[nearest_idx])


    if (len(cam_traj_global.keys()) != len(gt_ts)):
        print("Camera vs Timestamp counts differ!")
        print("\t", len(cam_traj_global.keys()), len(gt_ts))
        #sys.exit()

    obj_vels = obj_poses_to_vels(obj_traj, gt_ts)
    cam_vels = cam_poses_to_vels(cam_traj_global, gt_ts)
    
    #sys.exit()

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
        
        T_x = [vel[0][0] for vel in vels]
        T_y = [vel[0][1] for vel in vels]
        T_z = [vel[0][2] for vel in vels]
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


    plt.savefig(os.path.join(args.base_dir, 'velocity_plots.png'), dpi=600, bbox_inches='tight')
    #plt.show()
    #exit(0)

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

    #slice_dir = os.path.join(args.base_dir, 'slices_' + str(int(args.width * 100)).rjust(2, '0'))
    #vis_dir   = os.path.join(args.base_dir, 'vis_' + str(int(args.width * 100)).rjust(2, '0'))

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

    gt_ts[:] += args.offset

    cam_vel_file = open(os.path.join(args.base_dir, 'cam_vels_local_frame.txt'), 'w')
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

        if (len(nums) > i):
            cam_vel_file.write('frame_' + str(i).rjust(10, '0') + '.png' + " " +
                                            vel2text(cam_vels[nums[i]]) + "\n")

        nmin = np.nanmin(depth)
        nmax = np.nanmax(depth)

        #print (np.nanmin(depth), np.nanmax(depth))
        eimg[:,:,2] = (depth - nmin) / (nmax - nmin) * 255

        col_mask = mask_to_color(mask)
        col_mask += np.dstack((cimg, cimg * 0, cimg * 0)) * 1.0


        if (add_rgb):
            rgb_img = cv2.imread(os.path.join(rgb_dir, rgb_name_list[i].split('/')[-1]), cv2.IMREAD_COLOR)
            rgb_img = undistort_img(rgb_img,  K, D)

            rgb_img[mask > 10] = rgb_img[mask > 10] * 0.5 + col_mask[mask > 10] * 0.5
            eimg = np.hstack((rgb_img, eimg))
        else:
            eimg = np.hstack((eimg, col_mask))


        if (len(oids) > 0):
            col_vel = vel_to_color(mask, obj_vels[nums[i]])
            eimg = np.hstack((eimg, col_vel))

            footer = gen_text_stub(eimg.shape[1], cam_vels[nums[i]], obj_traj[nums[i]], obj_vels[nums[i]])
            eimg = np.vstack((eimg, footer))

        if (len(nums) > i and len(oids) == 0):
            footer = gen_text_stub(eimg.shape[1], cam_vels[nums[i]], None, None)
            eimg = np.vstack((eimg, footer))

        cv2.imwrite(os.path.join(vis_dir, 'frame_' + str(i).rjust(10, '0') + '.png'), eimg)

    cam_vel_file.close()
