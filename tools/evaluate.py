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


def read_inference(folder):
    print ("Reading inference in", folder)
    fnames = os.listdir(folder)
    
    depths = {}
    pposes = {}
    masks = {}
    egomotion = {}

    for fname in fnames:
        if ('.' not in fname):
            continue

        ext = fname.split('.')[-1]
        name = fname.split('.')[-2]
        if (ext != 'npy'): continue
        num = int(name.split('_')[-1])
       
        fname_ = os.path.join(folder, fname)
        #if ('pixel_pose' in name):
        #    pposes[num] = np.load(fname_)

        if ('residual_3d_pose' in name):
            pposes[num] = np.load(fname_)

        if ('motion_mask' in name):
            masks[num] = np.load(fname_)

        if ('ego_pose' in name):
            egomotion[num] = np.load(fname_)

        if ('depth_frame' in name):
            depths[num] = np.load(fname_)
   
    print ("Read residual: ", len(pposes), "masks", len(masks), 
           "egomotion", len(egomotion), "depth", len(depths))


    print ("Dataset read")
    return depths, egomotion, pposes, masks


# ====================================
def fixup_pose(gt, inf):
    gt_T = gt[0]
    gt_q = gt[1]

    inf_T = np.array([inf[2], -inf[0], -inf[1]])
    inf_RPY = np.array([inf[5], -inf[3], -inf[4]])

    gt_norm  = np.linalg.norm(gt_T)
    inf_norm = np.linalg.norm(inf_T)

    alpha = 1
    if (inf_norm > 0.0000001):
        alpha = gt_norm / inf_norm

    inf_T *= alpha
    return gt, [inf_T, inf_RPY]


def fixup_pposvel(ppos):
    cmb = np.zeros(ppos.shape, dtype=np.float32)
    cmb[:,:,1] =  ppos[:,:,2]
    cmb[:,:,0] = -ppos[:,:,0]
    cmb[:,:,2] = -ppos[:,:,1]

    if (ppos.shape[2] == 3):
        return cmb

    cmb[:,:,3] =  ppos[:,:,5]
    cmb[:,:,4] = -ppos[:,:,3]
    cmb[:,:,5] = -ppos[:,:,4]
    
    return cmb


def get_th_mask(ppos, bmask, th=0.2):
    ppos_th = np.copy(ppos)
    mask = np.ones(bmask.shape, dtype=np.float32)
    ppos_th[bmask <= th] = np.nan
    mask[bmask <= th] = 0
    norms = ppos_sqnorm(ppos_th)
    mean = np.nanmean(norms)
    mask[norms < mean / 4] = 0
    return mask


def th_ppos_mask(ppos, bmask):
    ppos_th = np.copy(ppos)
    ppos_th[bmask <= 0.5] = np.nan
    #ppos_th *= np.dstack((bmask, bmask, bmask))
    norms = ppos_sqnorm(ppos_th)
    mean = np.nanmean(norms)
    ppos_th[norms < mean / 4] = np.nan
    return ppos_th


def get_object_pos(bmask, ppos, gt_pos):
    ppos_th = th_ppos_mask(ppos, bmask)
    pos = np.nanmean(ppos_th, axis=(0, 1))
    inf_T = np.array([pos[0], pos[1], pos[2]])

    gt_norm  = np.linalg.norm(gt_pos[0])
    inf_norm = np.linalg.norm(inf_T)

    alpha = 1
    if (inf_norm > 0.0000001):
        alpha = gt_norm / inf_norm

    inf_T *= alpha
    return inf_T



CAM_AEE = 0.0
CAM_AEE_CNT = 0.0

CAM_AEE_ang = 0.0
CAM_AEE_ang_CNT = 0.0
def do_camera_ego(gt_, inf_):
    gt, inf = fixup_pose(gt_, inf_)

    global CAM_AEE, CAM_AEE_CNT, CAM_AEE_ang, CAM_AEE_ang_CNT
    CAM_AEE_CNT += 1
    EE, tf = get_EE(gt[0], inf[0])
    CAM_AEE += EE


    # normalized vectors at this point
    img_gt  = vel_to_color_splitmask({}, {}, gt)
    img_inf = vel_to_color_splitmask({}, {}, inf)
    return np.hstack((img_gt, img_inf))


DEPTH_CNT = 0.0
DEPTH_iRMSE = 0.0
DEPTH_SILog = 0.0
DEPTH_Th1 = 0.0
DEPTH_Th2 = 0.0
DEPTH_Th3 = 0.0
DEPTH_ARelDiff = 0.0
DEPTH_SqRelDiff = 0.0
DEPTH_linRMSE = 0.0
DEPTH_logRMSE = 0.0
DEPTH_L2_depth = 0.0
DEPTH_MSE = 0.0

def do_depth(gt, inf):
    shape = gt.shape
    gt[gt < 500] = np.nan
    gt /= 1000

    mask_nan = np.ones(shape)
    mask_nan[np.isnan(gt)] = 0

    #mask_close = np.ones(shape)
    #depth_inf_th = inf * np.nanmedian(gt) / np.nanmedian(inf)
    #depth_inf_th_rcp = depth_rcp(depth_inf_th)
    #mask_close[depth_inf_th_rcp > 150] = 0

    mask = mask_nan

    depth_bias = calculate_depth_bias(gt, inf, mask)
    depth_inf = inf * depth_bias

    depth_gt_rcp  = depth_rcp(gt)
    depth_inf_rcp = depth_rcp(depth_inf)

    iRMSE = calculate_depth_iRMSE(depth_gt_rcp, depth_inf_rcp, mask)
    SILog = calculate_depth_SILog(gt, depth_inf, mask) 
    Th1, Th2, Th3 = calculate_depth_Acc(gt, depth_inf, mask) 
    ARelDiff, SqRelDiff = calculate_depth_RelDiff(gt, depth_inf, mask)
    linRMSE, logRMSE = calculate_depth_RMSE(gt, depth_inf, mask)
    L2_depth, MSE = calculate_depth_L2(gt, depth_inf, mask)

    global DEPTH_CNT, DEPTH_iRMSE, DEPTH_SILog, DEPTH_Th1, DEPTH_Th2, DEPTH_Th3
    global DEPTH_ARelDiff, DEPTH_SqRelDiff, DEPTH_linRMSE, DEPTH_logRMSE, DEPTH_L2_depth, DEPTH_MSE

    DEPTH_CNT += 1
    DEPTH_iRMSE += iRMSE
    DEPTH_SILog += SILog
    DEPTH_Th1 += Th1
    DEPTH_Th2 += Th2
    DEPTH_Th3 += Th3
    DEPTH_ARelDiff += ARelDiff
    DEPTH_SqRelDiff += SqRelDiff
    DEPTH_linRMSE += linRMSE
    DEPTH_logRMSE += logRMSE
    DEPTH_L2_depth += L2_depth
    DEPTH_MSE += MSE

    depth_gt_rcp *= mask
    #depth_inf_rcp *= mask

    depth_gt3  = to3ch(depth_gt_rcp) / 4
    depth_inf3 = to3ch(depth_inf_rcp) / 4

    #depth_gt3[:,:,0] += t
    #depth_inf3[:,:,0] += t

    return np.hstack((depth_gt3, depth_inf3))



def cluster_pposes(inf_ppose, inf_mask):
    pass


OBJ_AOU = 0.0
OBJ_AOU_CNT = 0.0

OBJ_AEE = 0.0
OBJ_AEE_CNT = 0.0
def do_ppose(obj_poses, imask, inf_ppose, inf_mask):
    oids = sorted(obj_poses.keys())
    masks = mask_to_masks(imask, obj_poses)
    inf_ppose = fixup_pposvel(inf_ppose) * (-1)

    inf_T = inf_ppose[:,:,0:3]

    AEE_local = 0
    cnt = 0
    for oid in oids:
        gt_obj_0_p = obj_poses[oid]
        obj_0_p = get_object_pos(masks[oid], inf_T, gt_obj_0_p)
        EE, tf = get_EE(gt_obj_0_p[0], obj_0_p)
        print ("\t", tf, obj_0_p, gt_obj_0_p[0], obj_0_p - gt_obj_0_p[0])
        if (not np.isnan(EE)):
            cnt += 1
            AEE_local += EE

        new_op = np.copy(gt_obj_0_p[0])
        for i in range(3):
            new_op[i] = gt_obj_0_p[0][tf[0][i]] * tf[1][i]

        obj_poses[oid][0] = new_op


    global OBJ_AEE, OBJ_AEE_CNT, OBJ_AOU, OBJ_AOU_CNT
    if (cnt > 0):
        OBJ_AEE_CNT += 1
        OBJ_AEE += AEE_local / cnt

        AEE_local /= cnt
        print("\t", AEE_local)


    inf_bmask = np.zeros((inf_T.shape[0], inf_T.shape[1]), dtype=np.float32)
    if (len(inf_mask.shape) >= 3):
        inf_bmask = 1 - inf_mask[:,:,0]
        #for i in range(1, inf_mask.shape[2]):
        #    inf_bmask = np.hstack((inf_bmask, inf_mask[:,:,i]))
    else:
        inf_bmask = inf_mask


    obj_0_img = np.zeros(inf_T.shape, dtype=np.float32)
    gt_bin_mask = np.zeros(inf_T.shape, dtype=np.float32)
    for oid in oids:
        #th_pp = th_ppos_mask(inf_T, masks[oid])
        #th_pp[np.isnan(th_pp)] = 0
        #obj_0_img += th_pp
        #obj_0_img += inf_T * 
        gt_bin_mask +=  np.dstack((masks[oid], masks[oid], masks[oid]))

    obj_0_img = inf_T * np.dstack((inf_bmask, inf_bmask, inf_bmask))    
    inf_th_mask = get_th_mask(inf_T, inf_bmask, 0.3)

    iou_score = IOU(gt_bin_mask[:,:,0], inf_th_mask)
    if (not np.isnan(iou_score)):
        OBJ_AOU += iou_score
        OBJ_AOU_CNT += 1
        print("\tIOU:", iou_score)


    inf_bmask = np.dstack((inf_bmask, inf_bmask, inf_bmask)) * 255
    inf_th_mask = np.dstack((inf_th_mask, inf_th_mask, inf_th_mask)) * 255
    #obj_0_img = inf_T
    #inf_masks = cluster_pposes(inf_T, inf_mask)

    inf_T = obj_0_img
    lo = np.nanmin(inf_T)
    hi = np.nanmax(inf_T)

    img_inf = 255.0 * (inf_T - lo) / float(hi - lo)
    img_inf *= inf_bmask / 255

    #print (lo, hi, np.nanmax(img_inf))

    img_gt = vel_to_color(imask, obj_poses)
    #return np.hstack((img_gt, img_inf, gt_bin_mask * 255, inf_bmask, inf_th_mask))
    return np.hstack((img_gt, img_inf, inf_bmask))



def get_scores(inf_depth, inf_ego, inf_ppose, inf_mask,
               gt_depth, cam_vel, obj_poses, instance_mask):
    ego_img = do_camera_ego(cam_vel, inf_ego)
    depth_img = do_depth(gt_depth, inf_depth)
    ppose_img = do_ppose(obj_poses, instance_mask, inf_ppose, inf_mask)



    return np.hstack((depth_img, ppose_img))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        type=str,
                        default='.',
                        required=False)
    parser.add_argument('--inference_dir',
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

    inf_depths, inf_ego, inf_pposes, inf_masks = read_inference(args.inference_dir)

    # Trajectories
    cam_traj_global = read_camera_traj(os.path.join(args.base_dir, 'trajectory.txt'))
    obj_traj_global = read_object_traj(os.path.join(args.base_dir, 'objects.txt'))
    
    nums = sorted(obj_traj_global.keys())
    oids = sorted(obj_traj_global[nums[0]].keys())
    
    if (len(cam_traj_global.keys()) != len(obj_traj_global.keys())):
        print("Camera vs Obj pose numbers differ!")
        print("\t", len(cam_traj_global.keys()), len(obj_traj_global.keys()))
        sys.exit()

    obj_traj = to_cam_frame(obj_traj_global, cam_traj_global);

    print("Reading npz file...")
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
        sys.exit()

    print ("Compute velocities")
    obj_vels = obj_poses_to_vels(obj_traj, gt_ts)
    cam_vels = cam_poses_to_vels(cam_traj_global, gt_ts)

    slice_width = args.width

    #first_ts = cloud[0][0]
    #last_ts = cloud[-1][0]

    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    print (fx, fy, cx, cy) 

    print ("K and D:")
    print (K)
    print (D)
    print ("")

    vis_dir   = os.path.join(args.inference_dir, 'vis')
    clear_dir(vis_dir)

    #print ("The recording range:", first_ts, "-", last_ts)
    print ("The gt range:", gt_ts[0], "-", gt_ts[-1])
    print ("Discretization resolution:", discretization)
 
    for i, time in enumerate(gt_ts):
        #if (time > last_ts or time < first_ts):
        #    continue
        num = nums[i]
        if (i not in inf_depths.keys()):
            continue

        s = 0
        print ("\nComputing scores for frame", num, "index", i)
        #ev_img = get_scores(inf_depths[num], inf_ego[num], inf_pposes[num], inf_masks[num],
        #                    depth_gt[i], cam_vels[num], obj_vels[num], mask_gt[i])
        ev_img = get_scores(inf_depths[i], inf_ego[i], inf_pposes[i], inf_masks[i],
                            depth_gt[i - s], cam_vels[num - s], obj_vels[num - s], mask_gt[i - s])


        col_mask = mask_to_color(mask_gt[i])
        sl, _ = get_slice(cloud, idx, time, args.width, args.mode, discretization)
        eimg = dvs_img(sl, global_shape, K, D)
        #baseline_img = np.vstack((eimg, col_mask))
        ev_img = np.hstack((eimg, col_mask, ev_img))

        cv2.imwrite(os.path.join(vis_dir, 'frame_' + str(i).rjust(10, '0') + '.png'), ev_img)
        continue

        depth = depth_gt[i]
        mask  = mask_gt[i]

 
        cv2.imwrite(os.path.join(slice_dir, 'frame_' + str(i).rjust(10, '0') + '.png'), eimg)
        cv2.imwrite(os.path.join(slice_dir, 'depth_' + str(i).rjust(10, '0') + '.png'), depth.astype(np.uint16))
        cv2.imwrite(os.path.join(slice_dir, 'mask_'  + str(i).rjust(10, '0') + '.png'), mask.astype(np.uint16))

        nmin = np.nanmin(depth)
        nmax = np.nanmax(depth)

        #print (np.nanmin(depth), np.nanmax(depth))
        eimg[:,:,1] = (depth - nmin) / (nmax - nmin) * 255

        col_mask = mask_to_color(mask)
        col_vel = vel_to_color(mask, obj_vels[nums[i]])
        eimg = np.hstack((eimg, col_mask, col_vel))

        footer = gen_text_stub(eimg.shape[1], cam_vels[nums[i]], obj_traj[nums[i]], obj_vels[nums[i]])
        eimg = np.vstack((eimg, footer))

        cv2.imwrite(os.path.join(vis_dir, 'frame_' + str(i).rjust(10, '0') + '.png'), eimg)


    CAM_AEE_ang_CNT += 1

    print ("\n\n-----------------------")
    print ("Object AEE:", OBJ_AEE / OBJ_AEE_CNT)
    print ("Camera AEE:", CAM_AEE / CAM_AEE_CNT)
    print ("Camera AEE_ang:", CAM_AEE_ang / CAM_AEE_ang_CNT)
    print ("")
    print ("Object IOU:", OBJ_AOU / OBJ_AOU_CNT)
    print ("")
    print ("Depth inverse RMSE", DEPTH_iRMSE / DEPTH_CNT)
    print ("Depth linear  RMSE", DEPTH_linRMSE / DEPTH_CNT)
    print ("Depth log RMSE    ", DEPTH_logRMSE / DEPTH_CNT)
    print ("Depth MSE         ", DEPTH_MSE / DEPTH_CNT)
    print ("Depth SILog       ", DEPTH_SILog / DEPTH_CNT)
    print ("Depth Th1/2/3     ", DEPTH_Th1 / DEPTH_CNT, DEPTH_Th2 / DEPTH_CNT, DEPTH_Th3 / DEPTH_CNT)
    print ("Depth Abs Rel Diff", DEPTH_ARelDiff / DEPTH_CNT)
    print ("Depth Sq  Rel Diff", DEPTH_SqRelDiff / DEPTH_CNT)
    #print ("Depth L2 norm     ", DEPTH_L2_depth / DEPTH_CNT)
    print ("")
