#!/usr/bin/env python

import sys, os, shutil

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import yaml
import cv2
import OpenEXR
import Imath
import numpy as np
from math import fabs, sqrt
import pyquaternion as qt


# The dvs-related stuff, implemented in C. 
sys.path.insert(0, './build/lib.linux-x86_64-3.5') #The pydvs.so should be in PYTHONPATH!
import pydvs


class bcolors:
    HEADER = '\033[95m'
    PLAIN = '\033[37m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def offset(str_, p_offset):
    for i in range(p_offset):
        str_ = '...' + str_
    return str_

def hdr(str_, p_offset=0):
    return offset(bcolors.HEADER + str_ + bcolors.ENDC, p_offset)

def wht(str_, p_offset=0):
    return offset(bcolors.PLAIN + str_ + bcolors.ENDC, p_offset)

def okb(str_, p_offset=0):
    return offset(bcolors.OKBLUE + str_ + bcolors.ENDC, p_offset)

def okg(str_, p_offset=0):
    return offset(bcolors.OKGREEN + str_ + bcolors.ENDC, p_offset)

def wrn(str_, p_offset=0):
    return offset(bcolors.WARNING + str_ + bcolors.ENDC, p_offset)

def err(str_, p_offset=0):
    return offset(bcolors.FAIL + str_ + bcolors.ENDC, p_offset)

def bld(str_, p_offset=0):
    return offset(bcolors.BOLD + str_ + bcolors.ENDC, p_offset)


def ensure_dir(f):
    if not os.path.exists(f):
        print (okg("Created directory: ") + okb(f))
        os.makedirs(f)


def clear_dir(f):
    if os.path.exists(f):
        print (wrn("Removed directory: ") + okb(f))
        shutil.rmtree(f)
    os.makedirs(f)
    print (okg("Created directory: ") + okb(f))



global_scale_pn = 50
global_scale_pp = 50
global_shape = (260, 346)


def mask_to_color(mask):
    colors = [[255,255,0], [0,255,255], [255,0,255], 
              [0,255,0],   [0,0,255],   [255,0,0]]
     
    cmb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    m_ = np.max(mask) + 500
    i = 0
    while (m_ > 0):
        cmb[mask < m_] = np.array(colors[i % len(colors)])
        i += 1
        m_ -= 1000

    cmb[mask < 500] = np.array([0,0,0])
    return cmb




def vel_to_color_splitmask(obj_masks, obj_vel, bg_pos=None):
    oids = sorted(obj_vel.keys())
    shape = global_shape
    if (len(oids) > 0):
        shape = obj_masks[oids[0]].shape
    cmb = np.zeros((shape[0], shape[1], 3), dtype=np.float32)

    lo = 1
    rng = 100 
    if (True):
        all_vels = []
        if (bg_pos):
            all_vels.append(bg_pos[0])
        for id_ in oids:
            all_vels.append(obj_vel[id_][0])
        lo = -np.min(all_vels)
        rng = np.max(all_vels) + lo
        if (rng < 0.00001): rng = 1
        rng = 255.0 / rng

    if (bg_pos):
        cmb[:] = (lo + np.array([bg_pos[0][0], bg_pos[0][1], bg_pos[0][2]])) * rng

    for id_ in oids:
        vel = obj_vel[id_]
        obj_mask = obj_masks[id_]
        cmb[obj_mask == 1] = (lo + np.array([vel[0][0], vel[0][1], vel[0][2]])) * rng

    return cmb


def ppos_sqnorm(ppose):
    norm = (ppose[:,:,0] * ppose[:,:,0] +
            ppose[:,:,1] * ppose[:,:,1] +
            ppose[:,:,2] * ppose[:,:,2])
    return norm


def get_EE(v1, v2):
    d = v2 - v1
    return np.linalg.norm(d)


def mask_to_masks(mask, obj_vel):
    oids = sorted(obj_vel.keys())

    obj_masks = {}
    for id_ in oids:
        obj_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        cutoff_lo = id_ * 1000 - 100
        cutoff_hi = id_ * 1000 + 100
        obj_mask[np.where(np.logical_and(mask>=cutoff_lo, mask<=cutoff_hi))] = 1
        obj_masks[id_] = obj_mask
    return obj_masks


def vel_to_color(mask, obj_vel, bg_pos=None):
    obj_masks = mask_to_masks(mask, obj_vel)
    return vel_to_color_splitmask(obj_masks, obj_vel, bg_pos)


def undistort_img(img, K, D):
    Knew = K.copy()
    Knew[(0,1), (0,1)] = 0.87 * Knew[(0,1), (0,1)]
    img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
    return img_undistorted


def dvs_img(cloud, shape, K, D, slice_width=0.05):
    fcloud = cloud.astype(np.float32) # Important!

    cmb = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    pydvs.dvs_img(fcloud, cmb)
 
    cmb = undistort_img(cmb, K, D)
 
    cnt_img = cmb[:,:,0] + cmb[:,:,2] + 1e-8
    timg = cmb[:,:,1]
    
    timg[cnt_img < 0.99] = 0
    timg /= cnt_img

    cmb[:,:,0] *= global_scale_pp
    cmb[:,:,1] *= 255.0 / slice_width
    cmb[:,:,2] *= global_scale_pn

    return cmb
    return cmb.astype(np.uint8)


def get_slice(cloud, idx, ts, width, mode=0, idx_step=0.01):
    ts_lo = ts
    ts_hi = ts + width
    if (mode == 1):
        ts_lo = ts - width / 2.0
        ts_hi = ts + width / 2.0
    if (mode == 2):
        ts_lo = ts - width
        ts_hi = ts
    if (mode > 2 or mode < 0):
        print ("Wrong mode! Reverting to default...")
    if (ts_lo < 0): ts_lo = 0

    t0 = cloud[0][0]

    idx_lo = int((ts_lo - t0) / idx_step)
    idx_hi = int((ts_hi - t0) / idx_step)
    if (idx_lo >= len(idx)): idx_lo = -1
    if (idx_hi >= len(idx)): idx_hi = -1

    sl = np.copy(cloud[idx[idx_lo]:idx[idx_hi]].astype(np.float32))
    idx_ = np.copy(idx[idx_lo:idx_hi])

    if (idx_lo == idx_hi):
        return sl, np.array([0])

    if (len(idx_) > 0):
        idx_0 = idx_[0]
        idx_ -= idx_0

    if (sl.shape[0] > 0):
        t0 = sl[0][0]
        sl[:,0] -= t0

    return sl, idx_


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


def transform_pose(obj, cam):
    pos = obj[0] - cam[0]
    inv_rot = cam[1].inverse
    #inv_rot = cam[1]
    rotated_pos = inv_rot.rotate(pos)               
    return [rotated_pos, obj[1]]


def to_cam_frame(obj_traj_global, cam_traj_global):
    ret = {}
    nums = sorted(obj_traj_global.keys())
    oids = sorted(obj_traj_global[nums[0]].keys())

    for num in nums:
        ret[num] = {}
        for id_ in oids:
            curr_loc = transform_pose(obj_traj_global[num][id_], cam_traj_global[num])
            ret[num][id_] = curr_loc
    return ret


def compute_vel(p1, p2, dt):
    vel_t = (p2[0] - p1[0]) / dt
    vel_r = p2[1] * p1[1].inverse
    return [vel_t, vel_r]


def compute_vel_local(p1, p2, dt):
    p2_ = transform_pose(p2, p1)
    p1_ = [np.array([0, 0, 0]), qt.Quaternion(1, 0, 0, 0)]
    return compute_vel(p1_, p2_, dt)


def obj_poses_to_vels(obj_traj, gt_ts):
    ret = {}
    nums = sorted(obj_traj.keys())
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
    
    last_pos = cam_traj[nums[0]]
    last_t = gt_ts[0]
    for i, num in enumerate(nums):
        dt = gt_ts[i] - last_t
        if (dt < 0.0000001): dt = 1
        ret[num] = compute_vel_local(last_pos, cam_traj[num], dt) 
        last_pos = cam_traj[num]
        last_t = gt_ts[i]
    return ret

# ======================================================
# depth stuff
def depth_rcp(depth):
    #depth += 0.01
    img = np.reciprocal((depth + 0.01) / 300) * 1.5
    #img = np.clip(img, 0.001, 200)
    img[np.isnan(depth)] = 0
    return img

def to3ch(img):
    return np.dstack((img, img, img))


def calculate_depth_bias(d1, d2, mask):
    diff = d1 / d2
    diff[mask == 0] = np.nan
    diff[d2 == 0] = np.nan
    return np.nanmean(diff)

def calculate_depth_L2(d1, d2, mask):
    diff = np.square(d1 - d2)
    diff[mask == 0] = np.nan
    L2 = np.sqrt(diff)
    return L2, np.nanmean(diff)

def calculate_depth_iRMSE(p1, p2, mask):
    diff = np.square(p1 - p2)
    diff[mask == 0] = np.nan
    RMSE = np.sqrt(np.nanmean(diff))
    return RMSE

def calculate_depth_RMSE(d1, d2, mask):
    diff = np.square(d1 - d2)
    diff[mask == 0] = np.nan
    lin = np.sqrt(np.nanmean(diff))

    difflog = np.square(np.log(d2) - np.log(d1))
    difflog[d1 <= 0] = np.nan
    difflog[d2 <= 0] = np.nan
    difflog[mask == 0] = np.nan
    log = np.sqrt(np.nanmean(difflog))

    return lin, log

def calculate_depth_Acc(d1, d2, mask):
    delta = np.maximum(d1/d2, d2/d1)
    delta[d1 == 0] = np.nan
    delta[d2 == 0] = np.nan
    delta[mask == 0] = np.nan

    m1 = np.ones(d1.shape)
    m1[delta == np.nan] = 0
    n = np.sum(m1)
    
    m1[delta > 1.25**3] = 0
    t3 = np.sum(m1) / n

    m1[delta > 1.25**2] = 0
    t2 = np.sum(m1) / n

    m1[delta > 1.25**1] = 0
    t1 = np.sum(m1) / n
    return t1, t2, t3

def calculate_depth_RelDiff(d1, d2, mask):
    diff = np.abs(d1 - d2)
    diff[mask == 0] = np.nan
    diff[d1 == 0] = np.nan
    diff2 = np.square(diff)

    ab = np.nanmean(diff / d1)
    sq = np.nanmean(diff2 / d1)

    return ab, sq

# https://papers.nips.cc/paper/5539-depth-map-prediction-from-a-single-image-using-a-multi-scale-deep-network.pdf
# http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction
def calculate_depth_SILog(d1, d2, mask):
    di = np.log(d2) - np.log(d1)
    di[mask == 0] = np.nan
    di[d1 <= 0] = np.nan
    di[d2 <= 0] = np.nan
    di_2 = np.square(di)

    first_term = np.nanmean(di_2)
    second_term = np.nanmean(di)

    return first_term - second_term**2






# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

def matrix_from_quaternion(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)


""" Parse a dataset folder """
def parse_dataset(dataset_dir):
    
     # Parse camera calibration
    cam_file = open('%s/camera.yaml' % dataset_dir)
    cam_data = yaml.safe_load(cam_file)

    image_data = {}

    # Parse image paths       
    lines = [line.rstrip('\n') for line in open('%s/images.txt' % dataset_dir)]
    for line in lines:
        img_id, img_timestamp, img_path = line.split(' ')
        image_data[int(img_id)] = (float(img_timestamp), img_path)
    
     
    # Parse camera trajectory
    lines = [line.rstrip('\n') for line in open('%s/trajectory.txt' % dataset_dir)]
    for line in lines:
        splitted = line.split(' ')
        img_id = int(splitted[0])
        translation = [float(i) for i in splitted[1:4]]
        orientation = [float(i) for i in splitted[4:]]
        image_data[img_id] += (translation + orientation, )
        
    t = [frame[0] for frame in image_data.itervalues()]
    positions = [frame[2][:3] for frame in image_data.itervalues()]
    orientations = [frame[2][-4:] for frame in image_data.itervalues()]
    img_paths = [frame[1] for frame in image_data.itervalues()]
    
    width = cam_data['cam_width']
    height = cam_data['cam_height']
    fx = cam_data['cam_fx']
    fy = cam_data['cam_fy']
    cx = cam_data['cam_cx']
    cy = cam_data['cam_cy']
    
    cam = [width, height, fx, fy, cx, cy]
        
    return t, img_paths, positions, orientations, cam
   
   
class Frame:
    def __init__(self, frame_id, exr_path, use_log=True, blur_size=0, use_scharr=True):
        self.frame_id = frame_id
        self.exr_img = OpenEXR.InputFile(exr_path)
        self.img_raw = extract_grayscale(self.exr_img)
        
        # self.img is actually log(eps+img), blurred
        self.img = Frame.preprocess_image(self.img_raw.copy(), use_log=True, blur_size=blur_size)
        
        # self.img_raw is the non-logified image, blurred        
        self.img_raw = Frame.preprocess_image(self.img_raw, use_log=False, blur_size=blur_size)
        
        # compute the gradient using
        # nabla(log(eps+I)) = nabla(I) / (eps+I) (chain rule)
        # (hopefully better precision than directly
        # computing the numeric gradient of the log img)
        eps = 0.001
        self.gradient = compute_gradient(self.img_raw, use_scharr)
        self.gradient[:,:,0] = self.gradient[:,:,0] / (eps+self.img_raw)
        self.gradient[:,:,1] = self.gradient[:,:,1] / (eps+self.img_raw)
        self.z = extract_depth(self.exr_img)
    
    
    @staticmethod
    def preprocess_image(img, use_log=True, blur_size=0):
        if blur_size > 0:
            img = cv2.GaussianBlur(img, (blur_size,blur_size), 0)
            
        if use_log:
            img = safe_log(img)
        return img
        
        

class Trajectory:
    def __init__(self, times, positions, orientations):
        self.t = np.array(times)
        self.pos = np.array(positions)
        self.quat = np.array(orientations)
        
    
    def T_w_c(self, t):
        closest_id = self.find_closest_id(t)
        T_w_c = matrix_from_quaternion(self.quat[closest_id])
        T_w_c[:3,3] = self.pos[closest_id]
        return T_w_c
        
        
    def find_closest_id(self, t):
        idx = np.searchsorted(self.t, t, side="left")
        if fabs(t - self.t[idx-1]) < fabs(t - self.t[idx]):
            return idx-1
        else:
            return idx   
   
   
   
""" Log with a small offset to avoid problems at zero"""
def safe_log(img):
    eps = 0.001
    return np.log(eps + img)


""" Is pixel (x,y) inside a [width x height] image? (zero-based indexing) """
def is_within(x,y,width,height):
    return (x >= 0 and x < width and y >= 0 and y < height)

  
""" Return normalized vector """
def normalize(v):
    return v / np.linalg.norm(v)
   
   
""" Linear color space to sRGB
    https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_.28CIE_xyY_or_CIE_XYZ_to_sRGB.29 """
def lin2srgb(c):
    a = 0.055
    t = 0.0031308
    c[c <= t] = 12.92 * c[c <= t]
    c[c > t] = (1+a)*np.power(c[c > t], 1.0/2.4) - a
    return c


def extract_grayscale(img, srgb=False):
  dw = img.header()['dataWindow']

  size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
  precision = Imath.PixelType(Imath.PixelType.FLOAT)
  R = img.channel('R', precision)
  G = img.channel('G', precision)
  B = img.channel('B', precision)
  
  r = np.fromstring(R, dtype = np.float32)
  g = np.fromstring(G, dtype = np.float32)
  b = np.fromstring(B, dtype = np.float32)
  
  r.shape = (size[1], size[0])
  g.shape = (size[1], size[0])
  b.shape = (size[1], size[0])
  
  rgb = cv2.merge([b, g, r])
  grayscale = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
  
  if srgb:
      grayscale = lin2srgb(grayscale)

  return grayscale
  

def extract_bgr(img):
  dw = img.header()['dataWindow']

  size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
  precision = Imath.PixelType(Imath.PixelType.FLOAT)
  R = img.channel('R', precision)
  G = img.channel('G', precision)
  B = img.channel('B', precision)
  
  r = np.fromstring(R, dtype = np.float32)
  g = np.fromstring(G, dtype = np.float32)
  b = np.fromstring(B, dtype = np.float32)
  
  r.shape = (size[1], size[0])
  g.shape = (size[1], size[0])
  b.shape = (size[1], size[0])
  
  rgb = cv2.merge([b, g, r])
  return rgb
  

def extract_depth(img):
  dw = img.header()['dataWindow']
  size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
  precision = Imath.PixelType(Imath.PixelType.FLOAT)
  Z = img.channel('Z', precision)
  z = np.fromstring(Z, dtype = np.float32)
  z.shape = (size[1], size[0])
  return z
    

""" Compute horizontal and vertical gradients """
def compute_gradient(img, use_scharr=True):
    if use_scharr:
        norm_factor = 32
        gradx = cv2.Scharr(img, cv2.CV_32F, 1, 0, scale=1.0/norm_factor)
        grady = cv2.Scharr(img, cv2.CV_32F, 0, 1, scale=1.0/norm_factor)
    else:
        kx = cv2.getDerivKernels(1, 0, ksize=1, normalize=True)
        ky = cv2.getDerivKernels(0, 1, ksize=1, normalize=True)
        gradx = cv2.sepFilter2D(img, cv2.CV_32F, kx[0], kx[1])
        grady = cv2.sepFilter2D(img, cv2.CV_32F, ky[0], ky[1])
    
    gradient = np.dstack([gradx, grady])
    return gradient
