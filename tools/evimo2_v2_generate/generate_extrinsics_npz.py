#!/usr/bin/python3
import argparse
import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as Rot

def get_sequence_name(full_file_name):
    sequence_name = os.path.split(full_file_name)[1]
    return sequence_name

def get_camera_name(full_file_name):
    file_name_split = full_file_name.split(os.sep)
    camera_name = file_name_split[-4]
    return camera_name

def get_category(full_file_name):
    file_name_split = full_file_name.split(os.sep)
    category = file_name_split[-3]
    return category

def get_purpose(full_file_name):
    file_name_split = full_file_name.split(os.sep)
    purpose = file_name_split[-2]
    return purpose

def make_dir_if_needed(d):
    if not os.path.isdir(d):
        try:
            os.makedirs(d)
        except FileExistsError:
            print('Another process made {}'.format(d))

def read_camera_extrinsics(extrinsics_file):
    with open(extrinsics_file) as f:
        l = f.readline()
        l = l.rstrip('\n')
        items = l.split()
        extrinsics = [float(x) for x in items]

        f.readline() # skip a line
        cam_t_minus_pose_t = float(f.readline())

        other_t = float(f.readline())
        assert other_t == 0.0

    q = Rot.from_euler('xyz', extrinsics[3:], degrees=False).as_quat()

    return np.concatenate((extrinsics[:3], q)), cam_t_minus_pose_t

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stitch generated videos to together for website previews')
    parser = argparse.ArgumentParser()
    parser.add_argument('--idir', type=str, help='Directory containing raw folder tree')
    parser.add_argument('--ndir', type=str, help='Directory containing the npz folder tree')
    parser.add_argument('--odir', type=str, help='Directory containing the npz folder tree with only extrinsics data')
    args = parser.parse_args()

    raw_folder = args.idir
    npz_folder = args.ndir

    raw_glob = raw_folder + '/*/*/*'
    raw_folders = sorted(list(glob.glob(raw_glob)))

    ext_overlay_folder = args.odir
    make_dir_if_needed(ext_overlay_folder)

    for c in ['flea3_7', 'left_camera', 'right_camera', 'samsung_mono']:
        npz_glob = npz_folder + '/{}/*/*/*'.format(c)
        npz_folders = sorted(list(glob.glob(npz_glob)))

        for f in npz_folders:
            sequence_name = get_sequence_name(f)
            sequence_name_no_subseq_id = sequence_name[:-7]

            txt_ext_name = os.path.join(raw_folder,
                                        get_category(f),
                                        get_purpose(f),
                                        sequence_name_no_subseq_id,
                                        c,
                                       'extrinsics.txt')

            T_rc, _ = read_camera_extrinsics(txt_ext_name)

            ext_folder = os.path.join(ext_overlay_folder,
                                      c,
                                      get_category(f),
                                      get_purpose(f),
                                      sequence_name)
            make_dir_if_needed(ext_folder)

            ext_file = os.path.join(ext_folder, 'dataset_extrinsics.npz')

            t = {
                'x': T_rc[0],
                'y': T_rc[1],
                'z': T_rc[2]
            }

            q = {
                'x': T_rc[3],
                'y': T_rc[4],
                'z': T_rc[5],
                'w': T_rc[6]
            }

            np.savez_compressed(ext_file, t_rigcamera=t, q_rigcamera=q)
