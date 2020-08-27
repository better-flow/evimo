#!/usr/bin/python3

import numpy as np
import cv2
import glob
import yaml
import sys, os, math

import matplotlib.pyplot as plt

from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

import rigid_tf

class RigPoses:
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

    def calibrate(self, cam):
        cam.compute_relative_RT()
        print(len(cam.img_T), len(self.RPY))
        u0_arr = []
        u1_arr = []
        w_arr = []
        R0_arr = []
        R1_arr = []
        T0_arr = []
        T1_arr = []
        for i in range(0, len(self.RPY) - 1):
            if(i >= len(cam.img_T)): break
            for j in range(i + 1, len(self.RPY)):
                if(j >= len(cam.img_T)): break
                R_, t_ = cam.get_Rt(i, j)
                if (R_ is None or t_ is None): continue

                R, t = self.get_Rt(self.T[i], self.RPY[i], self.T[j], self.RPY[j])
                if (R is None or t is None): continue

                u0 = R.as_rotvec()
                u1 = R_.as_rotvec()
                e = np.abs(np.linalg.norm(u0) - np.linalg.norm(u1))
                if (e > 0.001): continue
                if (np.linalg.norm(u0) < 0.2): continue
                w_arr.append(e / np.linalg.norm(u0))
                u0 /= np.linalg.norm(u0)
                u1 /= np.linalg.norm(u1)

                print(i, j, ':', u0, np.linalg.norm(R.as_rotvec()), u1, np.linalg.norm(R_.as_rotvec()))
                print(t.transpose()[0], t_.transpose()[0], np.linalg.norm(t) - np.linalg.norm(t_))

                u0_arr.append(u0)
                u1_arr.append(u1)
                R0_arr.append(R)
                R1_arr.append(R_)
                T0_arr.append(t)
                T1_arr.append(t_)

        u0 = np.array(u0_arr)
        u1 = np.array(u1_arr)
        w = np.array(w_arr)# * 0 + 1
        w /= np.linalg.norm(w)

        R, rmsd = Rotation.align_vectors(u0, u1, weights=w)
        uerr = np.linalg.norm(u0 - (R.as_dcm() @ u1.transpose()).transpose(), axis=1)
        merr = np.median(uerr)
        print("Initial R:", R.as_euler('xyz'), "; median error:", merr)
        w[uerr > merr] = 0
        R, rmsd = Rotation.align_vectors(u0, u1, weights=w)
        print("Refined R:", R.as_euler('xyz'))

        a = np.zeros(shape=(0,3), dtype=np.float)
        b = np.zeros(shape=(0,1), dtype=np.float)
        for i in range(len(R0_arr)):
            R0 = R0_arr[i]
            R1 = R1_arr[i]
            T0 = T0_arr[i]
            T1 = T1_arr[i]

            a = np.vstack((a, R0.as_dcm() - np.eye(3)))
            b = np.vstack((b, R.as_dcm() @ T1 - T0))

        print (a.shape, b.shape)
        T, res, rnk, s = np.linalg.lstsq(a, b, rcond=None)

        print(R.as_dcm(), rmsd)
        print()
        print('R:', R.inv().as_rotvec())
        print(T.transpose()[0], res)
        print((R.as_dcm() @ T).transpose()[0])
        print((R.inv().as_dcm() @ T).transpose()[0])
        print((R.as_dcm() @ (-1 * T)).transpose()[0])
        print((R.inv().as_dcm() @ (-1 * T)).transpose()[0])
        print()
        print("x-y-z-R-P-Y:")
        print(np.hstack((T.transpose()[0], R.as_euler('xyz'))))
        return R.as_dcm(), T


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


class Camera:
    def __init__(self, calib, target, cb_config=None):
        self.name = calib['rostopic'].split('/')[1]
        K = calib['intrinsics']
        self.K = np.array([[K[0],0,K[2]],[0,K[1],K[3]],[0,0,1]], dtype=np.float32)
        self.D = np.array(calib['distortion_coeffs'], dtype=np.float32)
        self.res_x = calib['resolution'][0]
        self.res_y = calib['resolution'][1]
        self.dist_model = calib['distortion_model']
        self.tgt_rows = target['targetRows']
        self.tgt_cols = target['targetCols']
        print (self.name, ':\n', self.K, '\n', self.D, '\n', self.dist_model, '\n')

        self.objp = np.zeros((self.tgt_cols * self.tgt_rows, 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.tgt_cols,0:self.tgt_rows].T.reshape(-1,2)
        self.objp[:,0] *= target['colSpacingMeters']
        self.objp[:,1] *= target['rowSpacingMeters']

        self.images = []
        fnames = glob.glob(self.name + '/*.png')
        ts = [int(os.path.basename(fname).split('.')[0]) for fname in fnames]
        self.fnames = [n for _, n in sorted(zip(ts,fnames))]
        print (self.fnames)

        for fname in self.fnames:
            img = cv2.imread(fname)

            if (self.dist_model == 'radtan'):
                img = cv2.undistort(img, self.K, self.D, None, self.K)
            elif (self.dist_model == 'equidistant'):
                img = cv2.fisheye.undistortImage(img, self.K, self.D, None, self.K)
            else:
                print("Unknown undistortion model!")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.images.append(gray)

        self.img_R = []
        self.img_T = []
        #self.compute_relative_RT(cb_config)

    def compute_relative_RT(self, cb_config=None):
        self.img_R = []
        self.img_T = []
        self.marker_points = []
        for i, img in enumerate(self.images):
            ret, corners = cv2.findChessboardCorners(img, (self.tgt_cols, self.tgt_rows), None)
            if (not ret):
                self.img_R.append(None)
                self.img_T.append(None)
                continue
            img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(img, corners, (2,2), (-1,-1), criteria)
            ret_, rvecs, tvecs = cv2.solvePnP(self.objp, corners, self.K, np.zeros(4))

            R,_ = cv2.Rodrigues(rvecs)
            R = Rotation.from_matrix(R)
            self.img_R.append(R)
            self.img_T.append(tvecs)

            if (cb_config is not None):
                p_, _ = cv2.projectPoints(cb_config.reshape(-1,3), rvecs, tvecs, self.K, np.zeros(4))
                p_int = p_.reshape(-1,2).astype(np.int32)
                img_vis[p_int[:,1], p_int[:,0]] = np.array([0,0,255])
                self.marker_points.append(p_.reshape(-1,2))

            print ("Added", self.fnames[i])
            continue

            cv2.namedWindow('img', cv2.WINDOW_GUI_EXPANDED)
            axis = np.float32([[6,0,0], [0,6,0], [0,0,-3]]).reshape(-1,3) * target['rowSpacingMeters']
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, self.K, np.zeros(4))
            corner = tuple(corners[0].ravel())
            img_vis = cv2.line(img_vis, corner, tuple(imgpts[0].ravel()), (0,0,255), 1)
            img_vis = cv2.line(img_vis, corner, tuple(imgpts[1].ravel()), (0,255,0), 1)
            img_vis = cv2.line(img_vis, corner, tuple(imgpts[2].ravel()), (255,0,0), 1)
            img_vis = cv2.drawChessboardCorners(img_vis, (self.tgt_cols, self.tgt_rows), corners, ret)

            cv2.imshow('img', img_vis)
            cv2.waitKey(0)
            #break
        self.marker_points = np.array(self.marker_points, dtype=np.float32)


    def get_Rt(self, i0, i1, dcm=False):
        T0 = self.img_T[i0]
        T1 = self.img_T[i1]
        R0 = self.img_R[i0]
        R1 = self.img_R[i1]
        if (T0 is None or T1 is None):
            return None, None

        R0_inv = R0.inv()
        R = R1 * R0_inv

        T = T1 - np.expand_dims(R.apply(T0.transpose()[0]), axis=0).transpose()
        if (dcm): R = R.as_dcm()
        return R, T


    def plot_3d_markers(self, fname, rig, R, T):
        """
        Rt = np.array([[ 0.66311942, -0.51381591,  0.54430309],
             [-0.74838454, -0.44161449,  0.49487092],
             [-0.01390042, -0.73550653, -0.67737502]])
        Tt = np.array([[0.02214107], [-0.00067327], [-0.03020277]])

        Rr = np.array([[ 0.6650224,  -0.51521156,  0.54064984],
                      [-0.7467138,  -0.44630921,  0.49318009],
                      [-0.01279508, -0.7316865,  -0.6815212 ]])
        Tr = np.array([[-0.02110721], [ 0.00434272], [-0.03105385]])

        R = np.array([[ 0.66745093, -0.51308836,  0.53967545],
                      [-0.74455073, -0.44777291,  0.49511982],
                      [-0.01238817, -0.73228393, -0.68088676]])
        T = np.array([[-0.02324422], [ 0.00042172], [-0.03133713]])
        """

        f = open(fname, 'r')
        marker_poses = []
        for line in f.readlines():
            markers = line.split('|')[1:]
            pose_list = []
            for marker in markers:
                d = eval(marker)
                name = list(d.keys())[0]
                spl = d[name].split(' ')
                xyz = np.array([float(spl[0]), float(spl[1]), float(spl[2])])
                pose_list.append(xyz)
            marker_poses.append(pose_list)
        marker_poses = np.array(marker_poses)

        cv2.namedWindow('img', cv2.WINDOW_GUI_EXPANDED)
        for i, img in enumerate(self.images):
            img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            R_cam = Rotation.from_euler('xyz', rig.RPY[i], degrees=False)
            T_cam = rig.T[i]

            X1 = R_cam.as_dcm() @ (marker_poses[i].transpose() - T_cam)
            #X2 = np.linalg.inv(R) @ (X1 - T)
            X2 = np.linalg.inv(R) @ X1 + T

            p_ = self.K @ X2
            px = p_[0] / p_[2]
            py = p_[1] / p_[2]

            # plot
            px = px.astype(np.int32)
            py = py.astype(np.int32)
            mask = (px >= 0) & (py >= 0)
            mask &= (px < img_vis.shape[1]) & (py < img_vis.shape[0])
            px = px[mask]
            py = py[mask]

            img_vis[py, px] = np.array([0, 0, 255])

            cv2.imshow('img', img_vis)
            cv2.waitKey(0)

            #break

    def calibrate_3d(self, fname, rig):
        f = open(fname, 'r')
        marker_poses = []
        for line in f.readlines():
            markers = line.split('|')[1:]
            pose_list = []
            for i, marker in enumerate(markers):
                d = eval(marker)
                name = list(d.keys())[0]
                spl = d[name].split(' ')
                xyz = np.array([float(spl[0]), float(spl[1]), float(spl[2])])
                pose_list.append(xyz)
                if (i >= 2): break

            marker_poses.append(pose_list)
        marker_poses = np.array(marker_poses, dtype=np.float32)

        """
        self.marker_points[0,0] = np.array([475.9, 178.7])
        self.marker_points[0,1] = np.array([303.5, 375.1])
        self.marker_points[0,2] = np.array([1850, 1138.7]) / 3.0

        self.marker_points[1,0] = np.array([1416.8, 746.4]) / 3.0
        self.marker_points[1,1] = np.array([900.1, 1338.4]) / 3.0
        self.marker_points[1,2] = np.array([1856.9, 1363.6]) / 3.0

        self.marker_points[2,0] = np.array([1385.5, 47.7]) / 3.0
        self.marker_points[2,1] = np.array([868.7, 688.6]) / 3.0
        self.marker_points[2,2] = np.array([1790.7, 667.4]) / 3.0
        """

        p3d = []
        for i, img in enumerate(self.images):
            R_cam = Rotation.from_euler('xyz', rig.RPY[i], degrees=False)
            T_cam = rig.T[i]

            X1 = R_cam.as_dcm() @ (marker_poses[i].transpose() - T_cam)
            p3d.append(X1.transpose())
        p3d = np.array(p3d, dtype=np.float32)
        #p3d = p3d[:,1,:]
        #self.marker_points = self.marker_points[:,1,:]

        #p3d = p3d[:3,:,:]
        #self.marker_points = self.marker_points[:3,:,:]


        p3d = np.array(p3d).reshape(-1, 3).astype(np.float32)
        p_pix = self.marker_points.reshape(-1, 1, 2).astype(np.float32)

        print (p3d.shape, p_pix.shape)

        ret_, rvecs, tvecs = cv2.solvePnP(p3d, p_pix, self.K, np.zeros(4))
        #ret_, rvecs, tvecs, inliers = cv2.solvePnPRansac(p3d, p_pix, self.K, np.zeros(4),
        #                                                 reprojectionError=2.0,
        #                                                 iterationsCount=10000)
        #print (inliers)
        print (rvecs, tvecs)

        R = Rotation.from_rotvec(rvecs[:,0]).inv().as_dcm()
        print (R, tvecs)

        #X2 = np.linalg.inv(R) @ p3d.transpose() + tvecs
        p_, _ = cv2.projectPoints(p3d, rvecs, tvecs, self.K, np.zeros(4))

        err = self.marker_points.reshape(-1, 2) - p_.reshape(-1, 2)
        err = np.around(err, decimals=2)
        np.set_printoptions(suppress=True)
        print (self.marker_points)
        print (err)


        p_ = p_.reshape(-1, self.marker_points.shape[1], 2)
        cv2.namedWindow('img', cv2.WINDOW_GUI_EXPANDED)
        p_ *= 3
        self.marker_points *= 3
        for i, img in enumerate(self.images):
            img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_vis = cv2.resize(img_vis, dsize=(img_vis.shape[1] * 3, img_vis.shape[0] * 3))

            img_vis[self.marker_points[i,:,1].astype(np.int32), self.marker_points[i,:,0].astype(np.int32)] = np.array([255, 0, 0])
            img_vis[p_[i,:,1].astype(np.int32), p_[i,:,0].astype(np.int32)] = np.array([0, 0, 255])

            cv2.imshow('img', img_vis)
            cv2.waitKey(0)
            #plt.imshow(img_vis)
            #plt.show()

        return R, tvecs


    def calibrate_3d_II(self, fname, cb_config, rig):
        f = open(fname, 'r')
        marker_poses = []
        for line in f.readlines():
            markers = line.split('|')[1:]
            pose_list = []
            for i, marker in enumerate(markers):
                d = eval(marker)
                name = list(d.keys())[0]
                spl = d[name].split(' ')
                xyz = np.array([float(spl[0]), float(spl[1]), float(spl[2])])
                pose_list.append(xyz)
                if (i >= 2): break

            marker_poses.append(pose_list)
        marker_poses = np.array(marker_poses, dtype=np.float32)

        #cb_config[:] += np.array([0.04, 0.00, -0.0])


        cv2.namedWindow('img', cv2.WINDOW_GUI_EXPANDED)
        p3d = np.empty(shape=(0, 3), dtype=np.float32)
        p2d = np.empty(shape=(0, 1, 2), dtype=np.float32)
        img_outliers = set([])
        for i, img in enumerate(self.images):
            if (i not in set([10, 52])):
                img_outliers.add(i)
                continue

            img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            ret, corners = cv2.findChessboardCorners(img, (self.tgt_cols, self.tgt_rows), None)
            if (not ret):
                img_outliers.add(i)
                continue

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(img, corners, (2,2), (-1,-1), criteria)

            dst = marker_poses[i]
            src = cb_config.reshape(-1,3)
            R, T = rigid_tf.rigid_transform_3D(src.transpose(), dst.transpose())
            cb_corners_vicon = (R @ self.objp.transpose() + T).transpose()

            R_cam = Rotation.from_euler('xyz', rig.RPY[i], degrees=False)
            T_cam = rig.T[i]

            cb_corners_cam = (R_cam.as_dcm() @ (cb_corners_vicon.transpose() - T_cam)).transpose()

            p3d = np.vstack((p3d, cb_corners_cam))
            p2d = np.vstack((p2d, corners))

            ret_, rvecs, tvecs = cv2.solvePnP(cb_corners_cam, corners, self.K, np.zeros(4))
            #rvecs = np.array([[ 2.14115185, -0.9631415,   0.40938147]]).transpose()
            #rvecs = np.array([[ 2.14555134, -0.97130224,  0.41247228]]).transpose()
            #tvecs = np.array([[-0.02143744, -0.00691476, -0.03141196]]).transpose()

            print (i, ':', rvecs.transpose(), tvecs.transpose())

            p_, _ = cv2.projectPoints(cb_corners_cam, rvecs, tvecs, self.K, np.zeros(4))
            p_ = p_.reshape(-1, 2)

            img_vis[p_[:,1].astype(np.int32), p_[:,0].astype(np.int32)] = np.array([0, 0, 255])

            #cv2.imshow('img', img_vis)
            #cv2.waitKey(0)

        print (p3d.shape, p2d.shape)
        ret_, rvecs, tvecs = cv2.solvePnP(p3d, p2d, self.K, np.zeros(4))
        #ret_, rvecs, tvecs, inliers = cv2.solvePnPRansac(p3d, p2d, self.K, np.zeros(4),
        #                                                 reprojectionError=2.0,
        #                                                 iterationsCount=5000)

        print ("Total:", rvecs.transpose(), tvecs.transpose())
        #return


        # visualize
        for i, img in enumerate(self.images):
            if i in img_outliers:
                continue
            cv2.namedWindow('img_' + str(i), cv2.WINDOW_GUI_EXPANDED)


        T_err = (np.array([[0], [0], [0]], dtype=np.float) - 500) / 10000
        #rvecs = np.array([[2.14457984], [-0.96800249],  [0.40858797]])
        #tvecs = np.array([[0.02557283], [-0.0016614],  [-0.03080566]])
        #tvecs = np.array([[0.00113579], [-0.03358335],  [0.02183086]])
        #[0.01861063 0.01025673 0.03397296]
        #tvecs = np.array([[-0.00113579],  [0.03358335], [-0.02183086]])
        #tvecs = np.array([[-0.01861063], [-0.01025673], [-0.03397296]])
        #[-0.01963579  0.00118335 -0.04543086]

        rvecs_ = rvecs.copy()
        tvecs_ = tvecs.copy()
        def nothing(x):
            print (rvecs_.transpose()[0], tvecs_.transpose()[0], T_err.transpose()[0])
            pass
        cv2.createTrackbar('R','img',500,1000,nothing)
        cv2.createTrackbar('P','img',500,1000,nothing)
        cv2.createTrackbar('Y','img',500,1000,nothing)
        cv2.createTrackbar('x','img',500,1000,nothing)
        cv2.createTrackbar('y','img',500,1000,nothing)
        cv2.createTrackbar('z','img',500,1000,nothing)

        cv2.createTrackbar('R_','img',500,1000,nothing)
        cv2.createTrackbar('P_','img',500,1000,nothing)
        cv2.createTrackbar('Y_','img',500,1000,nothing)
        cv2.createTrackbar('x_','img',500,1000,nothing)
        cv2.createTrackbar('y_','img',500,1000,nothing)
        cv2.createTrackbar('z_','img',500,1000,nothing)



        while True:
            Rx = cv2.getTrackbarPos('R','img')
            Ry = cv2.getTrackbarPos('P','img')
            Rz = cv2.getTrackbarPos('Y','img')
            Tx = cv2.getTrackbarPos('x','img')
            Ty = cv2.getTrackbarPos('y','img')
            Tz = cv2.getTrackbarPos('z','img')

            Rx_ = cv2.getTrackbarPos('R_','img')
            Ry_ = cv2.getTrackbarPos('P_','img')
            Rz_ = cv2.getTrackbarPos('Y_','img')
            Tx_ = cv2.getTrackbarPos('x_','img')
            Ty_ = cv2.getTrackbarPos('y_','img')
            Tz_ = cv2.getTrackbarPos('z_','img')

            rvecs_ = rvecs + (np.array([[Rx], [Ry], [Rz]], dtype=np.float) - 500) / 10000
            tvecs_ = tvecs + (np.array([[Tx], [Ty], [Tz]], dtype=np.float) - 500) / 10000

            R_err = Rotation.from_euler('xyz', (np.array([Rx_, Ry_, Rz_], dtype=np.float) - 500) / 10000, degrees=False)
            T_err = (np.array([[Tx_], [Ty_], [Tz_]], dtype=np.float) - 500) / 10000

            for i, img in enumerate(self.images):
                if i in img_outliers:
                    continue
                img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                dst = marker_poses[i]
                src = cb_config.reshape(-1,3)
                R, T = rigid_tf.rigid_transform_3D(src.transpose(), dst.transpose())
                #print (R @ src.transpose() + T - dst.transpose())

                cb_corners_vicon = (R @ R_err.as_dcm() @ (self.objp.transpose() + T_err) + T).transpose()

                R_cam = Rotation.from_euler('xyz', rig.RPY[i], degrees=False)
                T_cam = rig.T[i]
                cb_corners_cam = (R_cam.as_dcm() @ (cb_corners_vicon.transpose() - T_cam)).transpose()

                p_, _ = cv2.projectPoints(cb_corners_cam, rvecs_, tvecs_, self.K, np.zeros(4))
                p_ = p_.reshape(-1, 2)
                p_ = np.vstack((p_, p_ + [0, 1], p_ + [1, 0], p_ + [1, 1]))

                mask = (p_[:,1] >= 0) & (p_[:,1] < img_vis.shape[0]) & (p_[:,0] >= 0) & (p_[:,0] < img_vis.shape[1])
                p_ = p_[mask]

                img_vis[p_[:,1].astype(np.int32), p_[:,0].astype(np.int32)] = np.array([0, 0, 255])

                cv2.imshow('img_' + str(i), img_vis)
            c = cv2.waitKey(30)
            if c == 27:
                break



    def sanity_markers(self, fname, cb_config, rig):
        f = open(fname, 'r')
        marker_poses = []
        for line in f.readlines():
            markers = line.split('|')[1:]
            pose_list = []
            for i, marker in enumerate(markers):
                d = eval(marker)
                name = list(d.keys())[0]
                spl = d[name].split(' ')
                xyz = np.array([float(spl[0]), float(spl[1]), float(spl[2])])
                pose_list.append(xyz)
                if (i >= 2): break

            marker_poses.append(pose_list)
        marker_poses = np.array(marker_poses, dtype=np.float32)

        self.compute_relative_RT()
        for i in range(0, len(rig.RPY) - 1):
            if(i >= len(self.img_T)): break
            for j in range(i + 1, len(rig.RPY)):
                if(j >= len(self.img_T)): break
                R_, T_ = self.get_Rt(i, j)
                if (R_ is None or T_ is None): continue

                R_cam_i = Rotation.from_euler('xyz', rig.RPY[i], degrees=False)
                T_cam_i = rig.T[i]
                R_cam_j = Rotation.from_euler('xyz', rig.RPY[j], degrees=False)
                T_cam_j = rig.T[j]

                dst = R_cam_i.as_dcm() @ (marker_poses[i].transpose() - T_cam_i)
                src = R_cam_j.as_dcm() @ (marker_poses[j].transpose() - T_cam_j)
                R, T = rigid_tf.rigid_transform_3D(src, dst)
                R = Rotation.from_matrix(R)

                u0 = R.as_rotvec()
                u1 = R_.as_rotvec()

                print (i, j, ':\t', T.transpose(), T_.transpose(), np.linalg.norm(T), np.linalg.norm(T_))
                print (i, j, ':\t', u0.transpose(), u1.transpose(), np.linalg.norm(u0), np.linalg.norm(u1))
                print ()


def get_marker_poses_local_frame(fname):
    try:
        f = open(fname, 'r')
    except:
        print ("Failed to open", fname)
        return None

    points = []
    for line in f.readlines():
        spl = line.split(' ')
        points.append(np.array([float(spl[0]), float(spl[1]), float(spl[2])]))
    return np.array(points)


target = yaml.load(open('target.yaml'), Loader=yaml.FullLoader)
calib  = yaml.load(open('camchain-output.yaml'), Loader=yaml.FullLoader)
rig_p = RigPoses('rig_0_poses.txt')
cb_config = get_marker_poses_local_frame('cb_0_config.txt')
for id_ in calib.keys():
    R = None
    T = None
    cam = Camera(calib[id_], target, cb_config)
    #R, T = rig_p.calibrate(cam)
    #R, T = cam.calibrate_3d('cb_0_poses.txt', rig_p)
    R, T = cam.calibrate_3d_II('cb_0_poses.txt', cb_config, rig_p)
    #cam.sanity_markers('cb_0_poses.txt', cb_config, rig_p)

    #cam.plot_3d_markers('cb_0_poses.txt', rig_p, R, T)
    break


cv2.destroyAllWindows()
