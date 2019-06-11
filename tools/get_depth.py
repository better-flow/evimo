#!/usr/bin/python3

import argparse
import numpy as np
import os, sys, math
import pyquaternion as qt
import pydvs, cv2


def read_camera_traj(folder_path):
    ret = {}
    
    print ("Reading camera trajectory:", folder_path)
    
    f = open(folder_path)
    for line in f.readlines():
        split = line.split(' ')
        num = int(split[0])
        
        v = np.array([float(split[1]),
                      float(split[2]),
                      float(split[3])])

        q = qt.Quaternion(float(split[4]),
                          float(split[5]),
                          float(split[6]),
                          float(split[7]))
        ret[num] = [v, q]
    f.close()

    if (len(ret) == 0):
        return {}

    minnum = min(ret.keys())
    maxnum = max(ret.keys())

    for i in range(minnum, maxnum + 1):
        if (i not in ret.keys()):
            print ("Error!", i, "not in range")
            sys.exit()

    print ("Read frames", minnum, "to", maxnum)
    return ret


def read_object_traj(folder_path):
    ret = {}
    keys = set()
    print ("Reading object trajectories:", folder_path)

    f = open(folder_path)
    for line in f.readlines():
        split = line.split(' ')
        num = int(split[0])
        id_ = int(split[1])

        v = np.array([float(split[2]),
                      float(split[3]),
                      float(split[4])])

        q = qt.Quaternion(float(split[5]),
                          float(split[6]),
                          float(split[7]),
                          float(split[8]))

        if (q.norm < 0.5):
            q = qt.Quaternion(1, 0, 0, 0)

        if (num not in ret.keys()):
            ret[num] = {}
        
        ret[num][id_] = [v, q]
        keys.add(id_)

    if (len(ret) == 0):
        return {}

    minnum = min(ret.keys())
    maxnum = max(ret.keys())

    for i in range(minnum, maxnum + 1):
        if (i not in ret.keys()):
            print ("Error!", i, "not in range")
            sys.exit()

    print("Object ids found:", keys)

    for num in ret.keys():
        if (ret[num].keys() != keys):
            print ("Error! frame", num)
            print ("\tids = ", ret[num].keys(), "expected", keys)
            sys.exit()

    f.close()
    return ret


def mask_to_color(mask):
    #colors = [[56,62,43], [26,50,63], [36,55,56], 
    #          [0,255,0],   [0,0,255],   [255,0,0]]

    colors = [[0,255,0],   [0,0,255],   [255,0,0]]

    cmb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    #cmb[:,:] = np.array([95, 96, 93])
    m_ = np.max(mask) + 500
    m_ = max(m_, 3500)
 
    maxoid = int(m_ / 1000)
    for i in range(maxoid):
        cutoff_lo = 1000.0 * (i + 1.0) - 5
        cutoff_hi = 1000.0 * (i + 1.0) + 5
        cmb[np.where(np.logical_and(mask>=cutoff_lo, mask<=cutoff_hi))] = np.array(colors[i % len(colors)])
    cmb *= 2.5

    return cmb


def quaternion_to_euler(q):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.degrees(math.atan2(t0, t1))
    #X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))
    #Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.degrees(math.atan2(t3, t4))
    #Z = math.atan2(t3, t4)

    return X, Y, Z 


def compute_vel(p1, p2, dt):
    # compute transform between frames instead of the velocity
    vel_t = (p2[0] - p1[0]) / 1.0 
    vel_r = p2[1] * p1[1].inverse
    return [vel_t, vel_r]


def transform_pose(obj, cam):
    pos = cam[0] - obj[0]
    #inv_rot = cam[1].inverse
    inv_rot = cam[1]
    rotated_pos = inv_rot.rotate(pos)
    return [rotated_pos, obj[1] * cam[1].inverse]


def compute_vel_local(p1, p2, dt):
    p2_ = transform_pose(p2, p1)
    p1_ = [np.array([0, 0, 0]), qt.Quaternion(1, 0, 0, 0)]
    return compute_vel(p1_, p2_, dt)


def obj_poses_to_vels(obj_traj, gt_ts):
    ret = {}
    nums = sorted(obj_traj.keys())

    if (len(nums) == 0):
        return {}

    oids = sorted(obj_traj[nums[0]].keys())

    for num in nums:
        ret[num] = {}

    for id_ in oids:
        last_pos = obj_traj[nums[0]][id_]
        last_t = gt_ts[0]
        for i, num in enumerate(nums):
            dt = gt_ts[i] - last_t
            if (dt < 0.0000001): dt = 1
            ret[num][id_] = compute_vel(last_pos, obj_traj[num][id_], dt) 
            last_pos = obj_traj[num][id_]
            last_t = gt_ts[i]

    return ret


def cam_poses_to_vels(cam_traj, gt_ts):
    ret = {}
    nums = sorted(cam_traj.keys())

    if (len(nums) == 0):
        return {}

    last_pos = cam_traj[nums[0]]
    last_t = gt_ts[0]
    for i, num in enumerate(nums):
        dt = gt_ts[i] - last_t
        if (dt < 0.0000001): dt = 1
        ret[num] = compute_vel_local(last_pos, cam_traj[num], dt) 
        last_pos = cam_traj[num]
        last_t = gt_ts[i]
    return ret


def vec_to_str(v, q=None, dt=None):
    ret = ""
    for elem in v:
        if (elem >= 0):
            ret += " "
        elem_ = elem
        if (dt is not None):
            elem_ /= dt
        ret += "{0:.2f}".format(elem_) + " "

    if (q is None):
        return ret

    ret += "/ "
    X, Y, Z = quaternion_to_euler(q)
    if (dt is not None):
        X /= dt
        Y /= dt
        Z /= dt

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


def gen_text_stub(shape_y, cam_vel, objs_pos, objs_vel, dt):
    oids = []
    if (objs_pos is not None):
        oids = objs_pos.keys()
    step = 20
    shape_x = (len(oids) + 2) * step + 10
    cmb = np.zeros((shape_x, shape_y, 3), dtype=np.float32)

    i = 0
    text = "Camera velocity = " + vec_to_str(cam_vel[0], cam_vel[1], dt)
    cv2.putText(cmb, text, (10, 20 + i * step), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    
    for id_ in oids:
        i += 1
        text = str(id_) + ": T = " + vec_to_str(objs_pos[id_][0]) + " | V = " + vec_to_str(objs_vel[id_][0], dt=dt)        
        cv2.putText(cmb, text, (10, 20 + i * step), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    return cmb


def dvs_img(cloud, shape, K, D, slice_width, mode=0):
    cmb = pydvs.dvs_img(cloud, shape, K=K, D=D)

    time = cmb[:,:,1]
    pcnt = cmb[:,:,2]
    ncnt = cmb[:,:,0]
    cnt = pcnt + ncnt

    # Scale up to be able to save as uint8
    cmb[:,:,0] *= 50
    cmb[:,:,1] *= 255.0 / slice_width
    cmb[:,:,2] *= 50

    if (mode == 1):
        cmb = np.dstack((time, pcnt, ncnt))

    return cmb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        type=str,
                        default='.',
                        required=False)
    parser.add_argument('--res',
                        nargs='+',
                        type=int,
                        required=False,
                        default=[260, 346])
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

    #obj_traj = to_cam_frame(obj_traj_global, cam_traj_global);
    obj_traj = obj_traj_global

    sl_npz = np.load(args.base_dir + '/recording.npz')
    cloud          = sl_npz['events']
    idx            = sl_npz['index']
    discretization = sl_npz['discretization']
    K              = sl_npz['K']
    D              = sl_npz['D']
    depth_gt       = sl_npz['depth']
    mask_gt        = sl_npz['mask']
    gt_ts          = sl_npz['gt_ts']

    slice_width = args.width
    first_ts = cloud[0][0]
    last_ts = cloud[-1][0]

    print ("The recording range:", first_ts, "-", last_ts)
    print ("The gt range:", gt_ts[0], "-", gt_ts[-1])
    print ("Discretization resolution:", discretization)
    print ("K and D:")
    print (K)
    print (D)
    print ("") 

    if (gt_ts[0] > 1.0):
        print("Time offset between events and image frames is too big:", gt_ts[0], "s.")
        gt_ts[:] -= gt_ts[0]

    gt_ts[:] += args.offset

    rgb_dir = os.path.join(args.base_dir, 'img')
    add_rgb = True
    if not os.path.exists(rgb_dir):
        add_rgb = False

    rgb_name_list = []
    if (add_rgb):
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

    # Compute velocities in local frame:
    obj_vels = obj_poses_to_vels(obj_traj, gt_ts)
    cam_vels = cam_poses_to_vels(cam_traj_global, gt_ts)

    # plot velocities / event rate
    import matplotlib.pyplot as plt
    plt.rcParams['lines.linewidth'] = 0.8

    # Event rate:
    erate = []
    for i, time in enumerate(gt_ts):
        if (time > last_ts or time < first_ts):
            continue
        sl, _ = pydvs.get_slice(cloud, idx, time, args.width, args.mode, discretization)
        erate.append(len(sl))
    erate = np.array(erate, dtype=np.float)
    erate /= np.mean(erate)

    # Compute dt for velocities:
    delta_t = [0]
    for i, ts in enumerate(gt_ts):
        if (i > 0): delta_t.append(ts - gt_ts[i - 1])
    delta_t[0] = delta_t[1]

    # Velocities:
    T_x = [cam_vels[i][0][0] / delta_t[i] for i in sorted(cam_vels.keys())]
    T_y = [cam_vels[i][0][1] / delta_t[i] for i in sorted(cam_vels.keys())]
    T_z = [cam_vels[i][0][2] / delta_t[i] for i in sorted(cam_vels.keys())]
    Euler = [quaternion_to_euler(cam_vels[i][1]) for i in sorted(cam_vels.keys())]
    Q_x = [q[0] / delta_t[i] for i, q in enumerate(Euler)]
    Q_y = [q[1] / delta_t[i] for i, q in enumerate(Euler)]
    Q_z = [q[2] / delta_t[i] for i, q in enumerate(Euler)]

    T_2 = np.array([T_x[i] * T_x[i] + T_y[i] * T_y[i] + T_z[i] * T_z[i] for i in range(len(T_x))])
    Q_2 = np.array([Q_x[i] * Q_x[i] + Q_y[i] * Q_y[i] + Q_z[i] * Q_z[i] for i in range(len(Q_x))])
    T_2 /= np.mean(T_2)
    Q_2 /= np.mean(Q_2)

    fig, axs = plt.subplots(2 * (len(oids) + 1), 1)

    #axs[0].plot(sorted(cam_vels.keys()), T_2)
    #axs[0].plot(sorted(cam_vels.keys()), erate, 'k--')
    #axs[1].plot(sorted(cam_vels.keys()), Q_2)
    #axs[1].plot(sorted(cam_vels.keys()), erate, 'k--')

    axs[0].plot(sorted(cam_vels.keys()), T_x, label='X axis (up - down)')
    axs[0].plot(sorted(cam_vels.keys()), T_y, label='Y axis (left - right)')
    axs[0].plot(sorted(cam_vels.keys()), T_z, label='Z axis (forward - backward)')
    axs[0].set_ylabel('camera linear (m/s)')
    axs[0].grid()
    axs[0].legend()
    axs[1].plot(sorted(cam_vels.keys()), Q_x, label='X axis')
    axs[1].plot(sorted(cam_vels.keys()), Q_y, label='Y axis')
    axs[1].plot(sorted(cam_vels.keys()), Q_z, label='Z axis')
    axs[1].set_xlabel('frame')
    axs[1].set_ylabel('camera angular (deg/s)')
    axs[1].grid()
    axs[1].legend()

    x_axis = [i for i in sorted(obj_vels.keys())]
    for k, id_ in enumerate(oids):
        vels = [obj_vels[i][id_] for i in x_axis]

        T_x = [vel[0][0] / delta_t[i] for vel in vels]
        T_y = [vel[0][1] / delta_t[i] for vel in vels]
        T_z = [vel[0][2] / delta_t[i] for vel in vels]
        Euler = [quaternion_to_euler(vel[1]) for vel in vels]
        Q_x = [q[0] / delta_t[i] for i, q in enumerate(Euler)]
        Q_y = [q[1] / delta_t[i] for i, q in enumerate(Euler)]
        Q_z = [q[2] / delta_t[i] for i, q in enumerate(Euler)]

        axs[2 * k + 2].plot(sorted(cam_vels.keys()), T_x, label='X axis')
        axs[2 * k + 2].plot(sorted(cam_vels.keys()), T_y, label='Y axis')
        axs[2 * k + 2].plot(sorted(cam_vels.keys()), T_z, label='Z axis')
        axs[2 * k + 2].set_ylabel('object_' + str(id_) + ' linear (m/s)')
        axs[2 * k + 2].grid()
        axs[2 * k + 2].legend()
        axs[2 * k + 3].plot(sorted(cam_vels.keys()), Q_x, label='X axis')
        axs[2 * k + 3].plot(sorted(cam_vels.keys()), Q_y, label='Y axis')
        axs[2 * k + 3].plot(sorted(cam_vels.keys()), Q_z, label='Z axis')
        axs[2 * k + 3].set_xlabel('frame')
        axs[2 * k + 3].set_ylabel('object_' + str(id_) + ' angular (deg/s)')
        axs[2 * k + 3].grid()
        axs[2 * k + 3].legend()

    fig.set_size_inches(0.03 * len(x_axis), 8 * (1 + len(oids)))
    plt.savefig(os.path.join(args.base_dir, 'velocity_plots.png'), dpi=200, bbox_inches='tight')
    #plt.show()

    # Generate images:
    slice_dir = os.path.join(args.base_dir, 'slices')
    vis_dir   = os.path.join(args.base_dir, 'vis')

    pydvs.replace_dir(slice_dir)
    pydvs.replace_dir(vis_dir)

    cam_vel_file = open(os.path.join(args.base_dir, 'cam_vels_local_frame.txt'), 'w')
    for i, time in enumerate(gt_ts):
        if (time > last_ts or time < first_ts):
            continue

        sl, _ = pydvs.get_slice(cloud, idx, time, args.width, args.mode, discretization)

        depth = depth_gt[i]
        mask  = mask_gt[i]
        eimg = dvs_img(sl, (args.res[0], args.res[1]), K, D, args.width, mode=0)

        cv2.imwrite(os.path.join(slice_dir, 'frame_' + str(i).rjust(10, '0') + '.png'), eimg)
        cv2.imwrite(os.path.join(slice_dir, 'depth_' + str(i).rjust(10, '0') + '.png'), depth.astype(np.uint16))
        cv2.imwrite(os.path.join(slice_dir, 'mask_'  + str(i).rjust(10, '0') + '.png'), mask.astype(np.uint16))

        # For visualization
        eimg = dvs_img(sl, (args.res[0], args.res[1]), K, D, args.width, mode=1)

        if (len(nums) > i):
            cam_vel_file.write('frame_' + str(i).rjust(10, '0') + '.png' + " " +
                                            vel2text(cam_vels[nums[i]]) + "\n")

        nmin = np.nanmin(depth)
        nmax = np.nanmax(depth)

        eimg[:,:,2] = (depth - nmin) / (nmax - nmin) * 255

        col_mask = mask_to_color(mask)

        if (add_rgb):
            rgb_img = cv2.imread(os.path.join(rgb_dir, rgb_name_list[i].split('/')[-1]), cv2.IMREAD_COLOR)
            rgb_img = pydvs.undistort_img(rgb_img, K, D)

            rgb_img[mask > 10] = rgb_img[mask > 10] * 0.5 + col_mask[mask > 10] * 0.5
            eimg = np.hstack((rgb_img, eimg))
        else:
            eimg = np.hstack((eimg, col_mask))

        if (len(oids) > 0):
            footer = gen_text_stub(eimg.shape[1], cam_vels[nums[i]], obj_traj[nums[i]], obj_vels[nums[i]], dt=delta_t[i])
            eimg = np.vstack((eimg, footer))

        if (len(nums) > i and len(oids) == 0):
            footer = gen_text_stub(eimg.shape[1], cam_vels[nums[i]], None, None, dt=delta_t[i])
            eimg = np.vstack((eimg, footer))

        cv2.imwrite(os.path.join(vis_dir, 'frame_' + str(i).rjust(10, '0') + '.png'), eimg)

    cam_vel_file.close()
