#!/usr/bin/python3

import numpy as np
import cv2
import glob
import yaml
import sys, os, math

from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

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
                print(t.transpose()[0], t_.transpose()[0])

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
        print(T.transpose()[0], res)
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
        self.compute_relative_RT(cb_config)

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
        Rt = np.array([[ 0.66311942, -0.51381591,  0.54430309],
             [-0.74838454, -0.44161449,  0.49487092],
             [-0.01390042, -0.73550653, -0.67737502]])
        Tt = np.array([[0.02214107], [-0.00067327], [-0.03020277]])

        R = np.array([[ 0.6650224,  -0.51521156,  0.54064984],
                      [-0.7467138,  -0.44630921,  0.49318009],
                      [-0.01279508, -0.7316865,  -0.6815212 ]])
        T = np.array([[-0.02110721], [ 0.00434272], [-0.03105385]])


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

        p3d = []
        for i, img in enumerate(self.images):
            R_cam = Rotation.from_euler('xyz', rig.RPY[i], degrees=False)
            T_cam = rig.T[i]

            X1 = R_cam.as_dcm() @ (marker_poses[i].transpose() - T_cam)
            p3d.append(X1.transpose())
        p3d = np.array(p3d, dtype=np.float32)
        #p3d = p3d[:,0,:]
        #self.marker_points = self.marker_points[:,0,:]

        p3d = np.array(p3d).reshape(-1, 3).astype(np.float32)
        p_pix = self.marker_points.reshape(-1, 1, 2).astype(np.float32)

        ret_, rvecs, tvecs = cv2.solvePnP(p3d, p_pix, self.K, np.zeros(4))
        print (rvecs, tvecs)

        R = Rotation.from_rotvec(rvecs[:,0]).inv().as_dcm()
        print (R, tvecs)

        #X2 = np.linalg.inv(R) @ p3d.transpose() + tvecs
        p_, _ = cv2.projectPoints(p3d, rvecs, tvecs, self.K, np.zeros(4))

        err = self.marker_points.reshape(-1, 2) - p_.reshape(-1, 2)
        err = np.around(err, decimals=2)
        np.set_printoptions(suppress=True)
        print (err)


        return R, tvecs


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
    R, T = cam.calibrate_3d('cb_0_poses.txt', rig_p)
    #R, T = rig_p.calibrate(cam)
    #cam.plot_3d_markers('cb_0_poses.txt', rig_p, R, T)
    break


cv2.destroyAllWindows()
