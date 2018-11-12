#!/usr/bin/python3

import argparse
import numpy as np
import cv2
import os, sys, signal, glob


def undistort_img(img, K, D):
    Knew = K.copy()
    Knew[(0,1), (0,1)] = 0.87 * Knew[(0,1), (0,1)]
    img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
    return img_undistorted


def read_calib(fname):
    print ("Reading camera calibration params from: ", fname)
    K = np.array([[0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]])
    D = np.array([0.0, 0.0, 0.0, 0.0])

    lines = []
    with open(fname) as calib:
        lines = calib.readlines()

    K_txt = lines[0:3]
    D_txt = lines[4]

    for i, line in enumerate(K_txt):
        for j, num_txt in enumerate(line.split(' ')[0:3]):
            K[i][j] = float(num_txt)

    for j, num_txt in enumerate(D_txt.split(' ')[0:4]):
        D[j] = float(num_txt)

    return K, D


def get_index(cloud, index_w):
    idx = [0]
    if (cloud.shape[0] < 1):
        return idx

    last_ts = cloud[0][0]
    for i, e in enumerate(cloud):
        while (e[0] - last_ts > index_w):
            idx.append(i)
            last_ts += index_w

    return np.array(idx, dtype=np.uint32)


def get_gt(path, K, D, base_dir):
    f = open(path)
    lines = f.readlines()
    f.close()

    num_lines = len(lines)
    print ("Number of ground truth samples:", num_lines)

    inm = lines[0].split(' ')[0]
    inm = os.path.join(base_dir, inm.split('/')[-1])

    print (inm)
    gt_img = cv2.imread(inm, cv2.IMREAD_UNCHANGED)
    print ("Gt shape:", gt_img.shape)
    print ("Gt type:", gt_img.dtype)

    ret = np.zeros((num_lines,) + (gt_img.shape[0], gt_img.shape[1]))

    ts = []
    for i, line in enumerate(lines):
        inm = line.split(' ')[0]
        inm = os.path.join(base_dir, inm.split('/')[-1])
        time = float(line.split(' ')[1]) 
        #gt_img = np.load(inm).astype(dtype=np.float32)
        gt_img = cv2.imread(inm, cv2.IMREAD_UNCHANGED).astype(dtype=np.float32)
        gt_img = undistort_img(gt_img, K, D)

        depth = gt_img[:,:,0]
        print (np.max(depth))

        depth[depth <= 10] = np.nan

        ts.append(time)
        ret[i,:,:] = depth

    return ret, np.array(ts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        type=str,
                        required=True)
    parser.add_argument('--discretization',
                        type=float,
                        required=False,
                        default=0.01)

    args = parser.parse_args()

    print ("Opening", args.base_dir)    

    K, D = read_calib(args.base_dir + '/calib.txt')
    print (K)
    print (D)

    print ("Reading the depth and flow")
    print ("Depth...")
    depth_gt, timestamps = get_gt(args.base_dir + '/ts.txt', K, D, args.base_dir + '/gt')

    print ("Reading the event file")
    cloud = np.loadtxt(args.base_dir + '/events.txt', dtype=np.float32)

    print ("Indexing")
    idx = get_index(cloud, args.discretization)


    print ("Saving...")
    np.savez_compressed(args.base_dir + "/recording.npz", events=cloud, index=idx, 
        discretization=args.discretization, K=K, D=D, depth=depth_gt, gt_ts=timestamps)

    print ("Done.")
