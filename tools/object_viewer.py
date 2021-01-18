#!/usr/bin/python3

import argparse
import numpy as np
import os, sys, math, time
from rigid_tf import *

def parse_settings(object_folder):
    ret = {}
    settings_lines = open(os.path.join(object_folder, 'settings.txt')).readlines()
    for l in settings_lines:
        spl = l.strip().split(':')
        if len(spl) != 2: continue
        ret[spl[0].strip()] = spl[1].strip()
    return ret

def parse_config(object_folder, settings):
    ret = {}
    settings_lines = open(os.path.join(object_folder, 'config.txt')).readlines()

    scale = 1.0
    if (settings['unit'] != 'mm'): scale = 1000.0

    for l in settings_lines:
        spl = l.strip().split()
        if len(spl) != 4: continue
        ret[int(spl[0])] = scale * np.array([float(spl[1]), float(spl[2]), float(spl[3])])
    return ret

def parse_vsk(object_folder):
    import xml.etree.ElementTree as ET
    root = ET.parse(os.path.join(object_folder, 'vicon.vsk')).getroot()

    object_vicon_name = ""
    vicon_marker_coord_map = {}
    for child in root:
        if (child.tag == 'Parameters'):
            for p in child:
                vicon_marker_coord_map[p.attrib['NAME']] = float(p.attrib['VALUE'])
        if (child.tag == 'Skeleton'):
            for p in child:
                object_vicon_name = p.attrib['NAME']

    ret = {}
    for key in vicon_marker_coord_map.keys():
        stripped_key = key.replace(object_vicon_name, '').replace('_', '')
        id_ = int(stripped_key[:-1])
        axis = stripped_key[-1]

        if id_ not in ret.keys():
            ret[id_] = np.array([0.0, 0.0, 0.0])

        if (axis == 'x'): ret[id_][0] = vicon_marker_coord_map[key]
        if (axis == 'y'): ret[id_][1] = vicon_marker_coord_map[key]
        if (axis == 'z'): ret[id_][2] = vicon_marker_coord_map[key]
    return ret

def transform_vicon_2_object(object_pts, vicon_pts):
    k0 = set(object_pts.keys())
    k1 = set(vicon_pts.keys())
    marker_ids = sorted(k0.intersection(k1))
    print ("vicon/object marker id intersection", marker_ids)

    obj_pt_npy = np.zeros(shape=(3, len(marker_ids)), dtype=np.float32)
    vcn_pt_npy = np.zeros(shape=(3, len(marker_ids)), dtype=np.float32)
    for i, mid in enumerate(marker_ids):
        obj_pt_npy[:,i] = object_pts[mid]
        vcn_pt_npy[:,i] = vicon_pts[mid]

    R, T = rigid_transform_3D(vcn_pt_npy, obj_pt_npy)
    return (R @ vcn_pt_npy + T).transpose(), obj_pt_npy.transpose(), marker_ids



def transform_object_2_vicon(object_pts, vicon_pts, obj, whitelist=None):
    k0 = set(object_pts.keys())
    k1 = set(vicon_pts.keys())
    if (whitelist is not None):
        k1 = k1.intersection(set(whitelist))
    marker_ids = sorted(k0.intersection(k1))
    print ("vicon/object marker id intersection", marker_ids)

    obj_pt_npy = np.zeros(shape=(3, len(marker_ids)), dtype=np.float32)
    vcn_pt_npy = np.zeros(shape=(3, len(marker_ids)), dtype=np.float32)
    for i, mid in enumerate(marker_ids):
        obj_pt_npy[:,i] = object_pts[mid]
        vcn_pt_npy[:,i] = vicon_pts[mid]

    R, T = rigid_transform_3D(obj_pt_npy, vcn_pt_npy)

    R_ = np.array([[ 0.903726,   -0.14523385, -0.40272397],
                  [-0.3248164,   0.38017777, -0.8660018 ],
                  [ 0.2788795,   0.9134397,   0.29640222]]) 

    T_ = np.array([[ 82.552795],
                  [ 12.040735],
                  [-14.74041 ]])
    R = np.linalg.inv(R_) @ R
    T -= T_

    print (R, T)
    obj_ = (R @ obj.transpose() + T).transpose()
    return vcn_pt_npy.transpose(), obj_, marker_ids


def generate_sphere(pos, r, samples=60):
    phi = np.linspace(0, np.pi, samples)
    theta = np.linspace(0, 2 * np.pi, 2 * samples)
    x = np.outer(np.sin(theta), np.cos(phi)).reshape(-1) * r + pos[0]
    y = np.outer(np.sin(theta), np.sin(phi)).reshape(-1) * r + pos[1]
    z = np.outer(np.cos(theta), np.ones_like(phi)).reshape(-1) * r + pos[2]
    return np.dstack((x,y,z))[0]

def estimate_marker_radii(object_folder, settings, config):
    import pyransac3d as pyrsc
    import open3d as o3d

    threshold = 0.4
    if (settings['unit'] != 'mm'): threshold /= 1000.0

    ret = {} 
    for marker_id in config.keys():
        ply_sphere_path = os.path.join(object_folder, str(marker_id) + '.ply')
        pcd = o3d.io.read_point_cloud(ply_sphere_path)
        points = np.asarray(pcd.points)

        sph = pyrsc.Sphere()
        center, radius, inliers = sph.fit(points, thresh=0.4)
        ret[marker_id] = radius
        print (marker_id, ':', center, radius)
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('object_folder',
                        type=str)
    args = parser.parse_args()

    print ("Opening", args.object_folder)

    settings = parse_settings(args.object_folder)
    print (settings)

    config = parse_config(args.object_folder, settings)
    print (config)

    vsk = parse_vsk(args.object_folder)
    print (vsk)

    radii = estimate_marker_radii(args.object_folder, settings, config)
    print ("\nRansac radii", radii)

    stock_radii = np.array([6.4, 7.9, 9.5, 12.7, 14.0, 15.9, 19.0]) / 2.0
    for mid in radii.keys(): # snap to stock
        best_fit_idx = np.argmin(np.abs(stock_radii - radii[mid]))
        radii[mid] = stock_radii[best_fit_idx]
    print ("Refined radii:", radii)


    import open3d as o3d
    object_cloud = o3d.io.read_point_cloud(os.path.join(args.object_folder, settings['mesh']))
    object_cloud.paint_uniform_color([1, 0.706, 0])

    #vicon_pts, obj_pts, marker_ids = transform_vicon_2_object(config, vsk)
    #whitelist=[1,2,6,5]
    whitelist=[4,3,7]
    vicon_pts, object_cloud_np_, marker_ids = transform_object_2_vicon(config, vsk, np.asarray(object_cloud.points), whitelist=whitelist)
    object_cloud.points = o3d.utility.Vector3dVector(object_cloud_np_)

    vicon_spheres = []
    for i, mid in enumerate(marker_ids):
        np_sphere = generate_sphere(vicon_pts[i], radii[mid])
        vicon_spheres.append(np_sphere)

    vicon_spheres = np.vstack(vicon_spheres)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vicon_spheres)
    o3d.io.write_point_cloud(os.path.join(args.object_folder, 'reference_markers.ply'), pcd)
    o3d.visualization.draw_geometries([pcd, object_cloud])
