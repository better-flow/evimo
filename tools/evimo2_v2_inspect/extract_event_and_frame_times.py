#!/usr/bin/python3
import argparse
import multiprocessing
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
    classical_frames = np.load(os.path.join(folder, 'dataset_classical.npz'), allow_pickle=True)
    if 'empty' in classical_frames:
        classical_frames_shape = None
        classical_frames == None
    else:
        classical_frames_shape = classical_frames[list(classical_frames.keys())[0]].shape

    depth = np.load(os.path.join(folder, 'dataset_depth.npz'))
    depth_shape = depth[list(depth.keys())[0]].shape

    meta = np.load(os.path.join(folder, 'dataset_info.npz'), allow_pickle=True)['meta'].item()
    frame_infos = meta['frames']

    depth_timestamps = np.array([frame_info['ts'] for frame_info in frame_infos])

    event_timestamps = np.load(os.path.join(folder, 'dataset_events_t.npy'), mmap_mode='r')

    save_timestamps(folder,
                    event_timestamps,
                    classical_frames_shape,
                    depth_shape,
                    depth_timestamps)

def test_polarity(folder):
    event_polarity = np.load(os.path.join(folder, 'dataset_events_p.npy'), mmap_mode='r')
    assert event_polarity.dtype == np.uint8
    assert len(event_polarity.shape) == 1
    assert np.all(event_polarity) < 2

def test_event_timestamps(folder):
    event_timestamps = np.load(os.path.join(folder, 'dataset_events_t.npy'), mmap_mode='r')

    if 'left_camera' in folder:
        camera_name = 'left_camera'
    elif 'right_camera' in folder:
        camera_name = 'right_camera'
    elif 'samsung_mono' in folder:
        camera_name = 'samsung_mono'
    elif 'flea3_7' in folder:
        camera_name = 'flea3_7'
    else:
        raise Exception('Camera name not found')

    if event_timestamps.shape[0] > 0:
        unique_times = np.unique(event_timestamps)
        dt = np.diff(unique_times)    
        min_dt = np.min(dt)
        max_dt = np.max(dt)
        avg_dt = np.average(dt)
        std_dt = np.std(dt)
        med_dt = np.median(dt)

        return camera_name, min_dt, max_dt, avg_dt, std_dt, med_dt
    return (camera_name,)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('idir',
                        type=str)
    parser.add_argument('odir',
                        type=str)
    args = parser.parse_args()

    output_dir = args.odir
    make_dir_if_needed(output_dir)

    folder = args.idir
    file_glob = folder + '/*/*/*/*'
    folders = sorted(list(glob.glob(file_glob)))

    with Pool(multiprocessing.cpu_count()) as p:
        list(tqdm(p.imap(extract_timestamps, folders), total=len(folders)))

    with Pool(multiprocessing.cpu_count()) as p:
        list(tqdm(p.imap(test_polarity, folders), total=len(folders)))

    with Pool(multiprocessing.cpu_count()) as p:
        list_camera_stats = list(tqdm(p.imap(test_event_timestamps, folders), total=len(folders)))

        stats_lists = {}
        for camera_name in ['left_camera', 'right_camera', 'samsung_mono']:
            if camera_name not in stats_lists:
                stats_lists[camera_name] = []

            for camera_stats in list_camera_stats:
                if camera_name == camera_stats[0]:
                    stats_lists[camera_name].append(camera_stats[1:])

        for camera_name in ['left_camera', 'right_camera', 'samsung_mono']:
            stats = np.array(stats_lists[camera_name])
            np.save(camera_name + '_stats', stats)
