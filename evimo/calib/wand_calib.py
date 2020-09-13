#!/usr/bin/python3


import os, sys, time, shutil, math
import numpy as np
import argparse
import cv2
import glob
import yaml

import matplotlib.pyplot as plt
import matplotlib.cbook as cbook


from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

import rigid_tf
import detect_wand


class Rig:
    def __init__(self, fname):
        f = open(fname, 'r')
        self.T = []
        self.RPY = []
        for line in f.readlines():
            spl = line.split(' ')
            tx = float(spl[1])
            ty = float(spl[2])
            tz = float(spl[3])
            qx = float(spl[4])
            qy = float(spl[5])
            qz = float(spl[6])
            qw = float(spl[7])

            # invert rotation to enforce X' = R(X - T)
            q = Quaternion(qw, qx, qy, qz).inverse
            self.RPY.append(self.q2rpy(q))
            self.T.append(np.array([[tx],[ty],[tz]]))

    # get R and T of the camera at time t=t1 in the frame of camera at t=t0
    def get_Rt(self, T0, RPY0, T1, RPY1, dcm=False):
        # looks like it performs 'zyx' order instead
        R0 = Rotation.from_euler('xyz', RPY0, degrees=False)
        R1 = Rotation.from_euler('xyz', RPY1, degrees=False)

        R0_inv = R0.inv()
        R = R1 * R0_inv # '*' instead of '@' because not numpy matrices

        T = np.expand_dims(R1.apply((T0 - T1).transpose()[0]), axis=0).transpose()
        if (dcm): R = R.as_dcm()
        return R, T

    def q2rpy(self, q):
        w = q[0]
        x = q[1]
        y = q[2]
        z = q[3]

        X = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        sin_val = 2.0 * (w * y - z * x)
        if (sin_val >  1.0): sin_val =  1.0
        if (sin_val < -1.0): sin_val = -1.0

        Y = math.asin(sin_val)
        Z = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return np.array([X, Y, Z])


class Wand:
    def __init__(self, fname):
        f = open(fname, 'r')
        marker_poses = []
        for line in f.readlines():
            markers = line.split('|')[1:]
            pose_list = []

            # assume that markers in the log appear in order
            for marker in markers:
                d = eval(marker)
                name = list(d.keys())[0]
                spl = d[name].split(' ')
                xyz = np.array([float(spl[0]), float(spl[1]), float(spl[2])])
                pose_list.append(xyz)
            marker_poses.append(pose_list)
        self.marker_poses = np.array(marker_poses)

    def to_rig_frame(self, rig, wand_3d_mapping):
        if (len(rig.RPY) != self.marker_poses.shape[0]):
            print ("Mismatch in rig and wand coordinates!", len(rig.RPY), "vs", self.marker_poses.shape[0])

        p3d = []
        mask = []
        print ("To rig frame svd errors (mm):")
        for i in range(len(rig.RPY)):
            R_cam = Rotation.from_euler('xyz', rig.RPY[i], degrees=False)
            X1 = R_cam.as_dcm() @ (self.marker_poses[i].transpose() - rig.T[i])

            R, T = rigid_tf.rigid_transform_3D(wand_3d_mapping['ir'].transpose(), X1)
            ir_X1  = R @ wand_3d_mapping['ir'].transpose() + T
            err = np.linalg.norm(X1 - ir_X1, axis=0) * 1e3
            print ('\t', i, err)

            if (np.max(err) > 1.0):
                print ("\t\trejected")
                mask.append(False)
            else:
                mask.append(True)
            red_X1 = R @ wand_3d_mapping['red'].transpose() + T
            p3d.append(red_X1.transpose())
            
            if (red_X1.transpose()[0,2] < 0):
                mask[-1] = False

        rig_points = np.array(p3d, dtype=np.float32)
        mask = np.array(mask)
        return rig_points, mask

    def detect(self, folder, wand_3d_mapping, th_rel=0.5, th_lin=0.5, th_ang=0.5, debug=False):
        self.images = []
        fnames = glob.glob(folder + '/*.png')
        ts = [int(os.path.basename(fname).split('.')[0]) for fname in fnames]
        self.fnames = [n for _, n in sorted(zip(ts,fnames))]
        print (self.fnames)

        detection_mask = []
        image_points = []
        blob_param = []
        for fname in self.fnames:
            image = cv2.imread(fname)
            keypoints = detect_wand.get_blobs(image)

            print ("\n\nProcessing", fname)
            print ("Keypoints:")
            print (keypoints)

            disp = None
            if (debug): disp = image
            idx, err = detect_wand.find_all_3lines(keypoints, th=max(image.shape[0], image.shape[1]) * 5e-3)
            wand_points, wand_idx = detect_wand.detect_wand(keypoints, idx, wand_3d_mapping, th_rel=th_rel, th_lin=th_lin, th_ang=th_ang, img_=disp)

            if (wand_points is None):
                detection_mask.append(False)
                wand_points = np.zeros(shape=(5, 2), dtype=np.float32)
                wand_idx    = np.zeros(shape=(5), dtype=np.int32)
                keypoints   = np.zeros(shape=(5, 3), dtype=np.float32)
            else:
                detection_mask.append(True)
            image_points.append(wand_points)
            blob_param.append(keypoints[wand_idx,2])

        return np.array(image_points), np.array(blob_param), np.array(detection_mask)


def plot_reprojection_error(detections, reprojected, p3d):
    print ("Plotting", p3d.shape, "points")
    distances = p3d[:,2]
    distances = 10 * (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

    fig, ax = plt.subplots(dpi=300)
    ax.scatter(detections[:,0], detections[:,1], c=None, s=distances**2, alpha=0.5)
    ax.scatter(reprojected[:,0], reprojected[:,1], c=None, s=distances**2, alpha=0.5)

    centroids = (detections + reprojected) / 2
    error = np.linalg.norm(detections - reprojected, axis=1)

    #ax.scatter(centroids[:,0], centroids[:,1], c=distances, s=error**2, alpha=0.5)

    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    ax.set_title('Reprojection error')

    ax.grid(True)
    fig.tight_layout()

    plt.show()



def estimate_individual(K, D, p3d, p_pix):
    ret_T = []
    ret_R = []
    for i in range(p3d.shape[0]):
        retval, rvecs, tvecs = cv2.solvePnP(p3d[i], p_pix[i], cameraMatrix=K, distCoeffs=D)
        ret_T.append(tvecs.transpose()[0])
        ret_R.append(rvecs.transpose()[0])

    ret_T = np.array(ret_T)
    ret_R = np.array(ret_R)

    values = np.hstack((ret_T, ret_R))
    md = np.median(values, axis=0)
    error = np.abs(values - md)

    th_80 = np.percentile(error, 90, axis=0)
    mask_Tx = error[:,0] < th_80[0]
    mask_Ty = error[:,1] < th_80[1]
    mask_Tz = error[:,2] < th_80[2]
    mask_Rx = error[:,3] < th_80[3]
    mask_Ry = error[:,4] < th_80[4]
    mask_Rz = error[:,5] < th_80[5]
    mask = mask_Tx & mask_Ty & mask_Tz & mask_Rx & mask_Ry & mask_Rz

    fig, ax = plt.subplots(2, 3, dpi=300, tight_layout=True)
    ax[0,0].hist(ret_T[:,0], bins=100)
    ax[0,0].hist(ret_T[mask,0], bins=100)
    ax[0,1].hist(ret_T[:,1], bins=100)
    ax[0,1].hist(ret_T[mask,1], bins=100)
    ax[0,2].hist(ret_T[:,2], bins=100)
    ax[0,2].hist(ret_T[mask,2], bins=100)
    ax[1,0].hist(ret_R[:,0], bins=100)
    ax[1,0].hist(ret_R[mask,0], bins=100)
    ax[1,1].hist(ret_R[:,1], bins=100)
    ax[1,1].hist(ret_R[mask,1], bins=100)
    ax[1,2].hist(ret_R[:,2], bins=100)
    ax[1,2].hist(ret_R[mask,2], bins=100)
    plt.show()

    return ret_T, ret_R, mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calibrate using Vicon calibration wand.')
    parser.add_argument('f', type=str, nargs='+', help='root dir(s)', default=[""])
    parser.add_argument('-c', type=str, help='camera name', default="cam_0")
    args = parser.parse_args()

    calib = yaml.load(open(os.path.join(args.f[0], args.c + '.yaml')), Loader=yaml.FullLoader)
    calib_ids = list(calib.keys())
    if (len(calib_ids) != 1):
        print ("Can only handel 1 camera per calib file!", calib_ids)
        sys.exit(-1)
    calib = calib[calib_ids[0]]
    K = calib['intrinsics']
    K = np.array([[K[0],0,K[2]],[0,K[1],K[3]],[0,0,1]], dtype=np.float32)
    D = np.array(calib['distortion_coeffs'], dtype=np.float32)
    res_x = calib['resolution'][0]
    res_y = calib['resolution'][1]
    dist_model = calib['distortion_model']

    print (args.c, ':\n', K, '\n', D, '\n', dist_model, '\n')
    if (dist_model != 'radtan' and dist_model != 'equidistant'):
        print ("Only 'radtan' and 'equidistant' models are supported!")
        sys.exit(-1)

    rig_points = []
    image_points = []
    blob_param = []
    mask = []
    for folder in args.f:
        npz_save_location = os.path.join(folder, "detections_" + args.c + ".npz")
        if os.path.exists(npz_save_location) and os.path.isfile(npz_save_location):
            print ("Using saved .npz file with detections:", npz_save_location)
            detections_npz = np.load(npz_save_location)
            rig_points_i = detections_npz['rig_points']
            image_points_i = detections_npz['image_points']
            blob_param_i = detections_npz['blob_param']
            mask_i = detections_npz['mask']
        else:
            rig = Rig(os.path.join(folder, 'rig_0_poses.txt'))
            wand = Wand(os.path.join(folder, 'wand_poses.txt'))
            wand_3d_mapping = {'red': np.array([[30.576542, -150.730270, -45.588951],
                                                [-45.614960, -18.425253, 2.359259],
                                                [-83.571022, 47.522499, 26.421692],
                                                [64.297485, 39.157578, 6.395612],
                                                [169.457520, 97.256927, 11.720362]]) * 1e-3,
                                'ir': np.array([[32.454151, -153.981064, -46.729141],
                                                [-43.735237, -21.440924, 1.182336],
                                                [-81.761475, 44.537903, 25.208614],
                                                [60.946213, 36.995384, 6.158191],
                                                [166.117447, 95.441071, 11.552736]]) * 1e-3}

            rig_points_i, mask1 = wand.to_rig_frame(rig, wand_3d_mapping)
            image_points_i, blob_param_i, mask2 = wand.detect(os.path.join(folder, args.c), wand_3d_mapping, th_rel=0.5, th_lin=0.5, th_ang=0.5)
            mask_i = mask1 & mask2
            np.savez(npz_save_location, mask=mask_i, rig_points=rig_points_i, image_points=image_points_i, blob_param=blob_param_i)

        rig_points.append(rig_points_i)
        image_points.append(image_points_i)
        blob_param.append(blob_param_i)
        mask.append(mask_i)

    rig_points = np.vstack(rig_points)
    image_points = np.vstack(image_points)
    blob_param = np.vstack(blob_param)
    mask = np.hstack(mask)

    print ("\nArray shapes:")
    print (rig_points.shape, rig_points[mask].shape)
    print (image_points.shape, image_points[mask].shape)

    p3d = rig_points[mask].reshape(1, -1, 3).astype(np.float32)
    p_pix = image_points[mask].reshape(1, -1, 2).astype(np.float32)
    blob_p = blob_param[mask].reshape(-1).astype(np.float32)

    print (p3d.shape, p_pix.shape, blob_p.shape)
    if (blob_p.shape[0] != p_pix.shape[1]):
        print ("Blob features are not aligned to pixel ids")
        sys.exit(-1)

    '''
    k = 0
    for i in range(rig_points[mask].shape[0]):
        for j in range(rig_points[mask].shape[1]):
            print('\n', p3d[0,k], '->', p_pix[0,k])
            print(rig_points[mask][i,j], '->',  image_points[mask][i,j])
            k += 1
    '''

    #_, _, mask_ind = estimate_individual(K, D, rig_points[mask], image_points[mask])
    #p3d = rig_points[mask][mask_ind].reshape(1, -1, 3).astype(np.float32)
    #p_pix = image_points[mask][mask_ind].reshape(1, -1, 2).astype(np.float32)
    #blob_p = blob_param[mask][mask_ind].reshape(-1).astype(np.float32)


    print ("\nRunning calibration")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(1e6), 1e-8)


    flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
    #flags += cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_ZERO_TANGENT_DIST
    #D *= 0
    #flags += cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2
    #flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL
    if (dist_model == 'radtan'):
        retval, K_, D_, rvecs, tvecs = cv2.calibrateCamera(p3d, p_pix, imageSize=(res_y, res_x),
                                                           cameraMatrix=K, distCoeffs=D,
                                                           flags=flags,
                                                           criteria=criteria)
        rvecs = rvecs[0]
        tvecs = tvecs[0]

        #retval, rvecs, tvecs = cv2.solvePnP(p3d, p_pix, cameraMatrix=K, distCoeffs=D)
        p_pix_reproj, _ = cv2.projectPoints(p3d[0], rvecs.reshape(1,3), tvecs.reshape(1,3), K_, D_)
    elif (dist_model == 'equidistant'):
        print ("Model", dist_model, "is not supported!")
        sys.exit(0)

        p3d__ = p3d.reshape(1, 1, -1, 3)
        #p3d__[0,0,:,0] = p3d[0,:,1]
        #p3d__[0,0,:,1] = p3d[0,:,0]

        retval, K_, D_, rvecs, tvecs = cv2.fisheye.calibrate(p3d__, p_pix.reshape(1, -1, 1, 2),
                                                             image_size=(res_y, res_x), K=K, D=D,
                                                             flags=flags)# + cv2.fisheye.CALIB_CHECK_COND)
        print (rvecs, tvecs)
        print (K, D)

        sys.exit(0)
        p_pix_reproj, _ = cv2.fisheye.projectPoints(p3d, rvecs[0], tvecs[0], K_, D_)
        print(p_pix_reproj)
    else:
        print ("Model", dist_model, "is not supported!")
        sys.exit(0)


    error = np.linalg.norm(p_pix_reproj[:,0] - p_pix[0], axis=1)
    print ("\tAverage error / reprojection error:", np.mean(error), retval)

    #for i in range(p_pix[0].shape[0]):
    #    print(p_pix[0,i], '->', p_pix_reproj[i,0], '\t', error[i])

    th = np.percentile(error, 30)
    print ("\tThresholding at", th)
    outlier_mask = error < th

    # Plot error histograms
    fig, ax = plt.subplots(2, 2, dpi=300, tight_layout=True)
    th_95 = np.percentile(error, 95)
    outlier_mask_95 = error < th_95
    ax[0,0].hist(error[outlier_mask_95], bins=100)
    ax[0,1].hist(p3d[0,:,2], bins=100)
    ax[0,1].hist(p3d[0,outlier_mask,2], bins=100)
    ax[1,0].hist(p3d[0,:,0], bins=100)
    ax[1,0].hist(p3d[0,outlier_mask,0], bins=100)
    ax[1,1].hist(p3d[0,:,1], bins=100)
    ax[1,1].hist(p3d[0,outlier_mask,1], bins=100)
    #ax[1,0].hist(blob_p, bins=100)
    #ax[1,0].hist(blob_p[outlier_mask], bins=100)

    #ax[1,0].hist(p_pix[0,:,0], bins=100)
    #ax[1,0].hist(p_pix[0,outlier_mask,0], bins=100)
    #ax[1,1].hist(p_pix[0,:,1], bins=100)
    #ax[1,1].hist(p_pix[0,outlier_mask,1], bins=100)
    #ax[1,0].hist2d(p_pix[0,:,0], p_pix[0,:,1], bins=100)
    #ax[1,1].hist2d(p_pix[0,outlier_mask,0], p_pix[0,outlier_mask,1], bins=100)
    plt.show()

    print (K_)
    print ("\nRunning calibration")
    retval, K_, D_, rvecs, tvecs = cv2.calibrateCamera(p3d[:,outlier_mask,:], p_pix[:,outlier_mask,:], imageSize=(res_y, res_x),
                                                       cameraMatrix=K_, distCoeffs=D_,
                                                       flags=flags,
                                                       criteria=criteria)
    rvecs = rvecs[0]
    tvecs = tvecs[0]

    #retval, rvecs, tvecs = cv2.solvePnP(p3d, p_pix, cameraMatrix=K, distCoeffs=D)

    p_pix_reproj, _ = cv2.projectPoints(p3d[0], rvecs.reshape(1,3), tvecs.reshape(1,3), K_, D_)
    error = np.linalg.norm(p_pix_reproj[:,0] - p_pix[0], axis=1)
    print ("\tAverage error / reprojection error:", np.mean(error[outlier_mask]), retval)

    p_pix = p_pix[:, outlier_mask]
    p3d = p3d[:, outlier_mask]
    p_pix_reproj = p_pix_reproj[outlier_mask]

    print (K_)
    print (D_)
    print (np.array(rvecs))
    print (tvecs)

    R = Rotation.from_rotvec(np.array(rvecs).reshape(3)).inv()
    T = np.array(tvecs).reshape(3, 1)
    T = -R.as_dcm() @ T

    print ("\nIntrinsic:")
    print (K_[0,0], K_[1,1], K_[0,2], K_[1,2], D_[0,0], D_[1,0], D_[2,0], D_[3,0])

    print ("\nExtrinsic:")
    print("x-y-z-R-P-Y:")
    print(np.hstack((T.reshape(3), R.as_euler('xyz'))))

    plot_reprojection_error(p_pix[0], p_pix_reproj[:,0], p3d[0])
