#import cv2
import numpy as np
import pprint
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import pickle
import os
from multiprocessing import Pool

def get_sequence_name(full_file_name):
    file_name = os.path.split(full_file_name)[1]
    sequence_name = os.path.splitext(file_name)[0]
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
            os.mkdir(d)
        except FileExistsError:
            print('Another process made {}'.format(d))

def save_timestamps(full_file_name,
                    event_timestamps,
                    classical_frames_shape,
                    depth_shape,
                    depth_timestamps):
    output_file = os.path.join(output_dir, get_camera_name(full_file_name))
    make_dir_if_needed(output_file)

    output_file = os.path.join(output_file, get_category(full_file_name))
    make_dir_if_needed(output_file)

    output_file = os.path.join(output_file, get_purpose(full_file_name))
    make_dir_if_needed(output_file)

    output_file = os.path.join(output_file, os.path.split(full_file_name)[1])

    if event_timestamps.shape != (0,):
        event_timestamps_first_last = np.array((event_timestamps[0], event_timestamps[-1]))
    else:
        event_timestamps_first_last = None

    np.savez(output_file,
             classical_frames_shape=classical_frames_shape,
             event_timestamps_shape=event_timestamps.shape,
             event_timestamps_first_last=event_timestamps_first_last,
             depth_shape=depth_shape,
             depth_timestamps=depth_timestamps)

def extract_timestamps(folder):
    try:
        classical_frames_shape = np.load(os.path.join(folder, 'classical.npy'), mmap_mode='r').shape
    except ValueError as e:
        # On event camera sequences it is a none object, make sure it is
        classical_frames = np.load(os.path.join(folder, 'classical.npy'), allow_pickle=True)
        classical_frames_shape = None
        assert classical_frames == None

    depth = np.load(os.path.join(folder, 'depth.npy'), mmap_mode='r')
    depth_shape = depth.shape

    meta = np.load(os.path.join(folder, 'meta.npy'), allow_pickle=True).item() #data['meta'].item()
    frame_infos = meta['frames']
    depth_timestamps = np.array([frame_info['cam']['ts'] for frame_info in frame_infos])

    events = np.load(os.path.join(folder, 'events.npy'), mmap_mode='r')
    event_timestamps = events[:, 0]

    save_timestamps(folder,
                    event_timestamps,
                    classical_frames_shape,
                    depth_shape,
                    depth_timestamps)

if __name__ == '__main__':
    output_dir = 'extracted_timestamps'
    make_dir_if_needed(output_dir)

    folder = '/media/levi/EVIMO/npz_extracted'
    file_glob = folder + '/*/*/*/*'
    files = sorted(list(glob.glob(file_glob)))

    with Pool(8) as p:
        list(tqdm(p.imap(extract_timestamps, files), total=len(files)))
