#!/usr/bin/python3
import argparse
import numpy as np
import pprint
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import pickle
import os
import multiprocessing
from multiprocessing import Pool

from extract_event_and_frame_times import (get_sequence_name,
                                           get_camera_name,
                                           get_category,
                                           get_purpose,
                                           make_dir_if_needed)

# https://stackoverflow.com/a/57364423
# istarmap.py for Python 3.8+
import multiprocessing.pool as mpp


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap


def split_timestamps_by_valid_periods(depth_timestamps, gap_detection_threshold):
    delta_t = depth_timestamps[1:] - depth_timestamps[:-1]
    gap_starts = delta_t >= gap_detection_threshold

    starts_ends_invalid_periods = []
    starts_ends_invalid_period = []
    valid_periods_timestamps = []
    valid_period_timestamps = []
    for i, t in enumerate(depth_timestamps):
        valid_period_timestamps.append(t)

        if i < len(depth_timestamps) - 1 and gap_starts[i] == True:
            valid_periods_timestamps.append(np.array(valid_period_timestamps))
            valid_period_timestamps = []
            starts_ends_invalid_periods.append(np.array((t, depth_timestamps[i+1])))

        if i == len(depth_timestamps) - 1:
            valid_periods_timestamps.append(np.array(valid_period_timestamps))
            valid_period_timestamps = []

    return valid_periods_timestamps, starts_ends_invalid_periods

def calc_depth_valid_stats(depth_timestamps, data_start_first_last, gap_detection_threshold):
    valid_periods_timestamps, starts_ends_invalid_periods = (
        split_timestamps_by_valid_periods(depth_timestamps, gap_detection_threshold))

    valid_ranges = []
    for timestamps in valid_periods_timestamps:
        valid_ranges.append(timestamps[-1] - timestamps[0])
    valid_ranges = np.array(valid_ranges)
    valid_total_range = np.sum(valid_ranges)

    #starts_ends_invalid_periods = find_start_end_invalid_periods(depth_timestamps, data_start_first_last, gap_detection_threshold)
    invalid_ranges = []
    for start_end in starts_ends_invalid_periods:
        invalid_ranges.append(start_end[1] - start_end[0])
    invalid_ranges = np.array(invalid_ranges)

    num_invalid_range  = invalid_ranges.shape[0]

    if invalid_ranges.shape[0] > 0:
        invalid_max_range = np.max(invalid_ranges)
    else:
        invalid_max_range = 0

    return num_invalid_range, valid_total_range, invalid_max_range

def analyze_timetamps(category, purpose, name, sequence_group, plot=True, id=0):
    timestamp_infos = {}

    for camera in sequence_group:
        timestamp_infos[camera] = np.load(sequence_group[camera], allow_pickle=True)

    num_event_depth_frames = None

    first_time = None
    last_time = None
    for camera in timestamp_infos:
        timestamp_info = timestamp_infos[camera]

        event_timestamps_first_last = timestamp_info['event_timestamps_first_last']

        if event_timestamps_first_last.shape != ():
            if first_time is None:
                first_time = event_timestamps_first_last[0]
                last_time = event_timestamps_first_last[1]
            else:
                if event_timestamps_first_last[0] < first_time:
                    first_time = event_timestamps_first_last[0]
                if event_timestamps_first_last[1] > last_time:
                    last_time = event_timestamps_first_last[1]

    if plot:
        plt.figure()

    for camera in timestamp_infos:
        timestamp_info = timestamp_infos[camera]

        event_timestamps_shape      = timestamp_info['event_timestamps_shape']
        event_timestamps_first_last = timestamp_info['event_timestamps_first_last']
        classical_frames_shape      = timestamp_info['classical_frames_shape']
        depth_shape                 = timestamp_info['depth_shape']
        depth_timestamps            = timestamp_info['depth_timestamps']

        if plot:
            if camera == 'flea3_7':
                plt.subplot(4,1,1)
                plt.plot(depth_timestamps[:-1], depth_timestamps[1:] - depth_timestamps[:-1])
                plt.xlim([first_time, last_time])
                plt.ylim([0, 3.0/60.0])
                plt.ylabel('{}'.format(camera))
            elif camera == 'left_camera':
                plt.subplot(4,1,2)
                plt.plot(depth_timestamps[:-1], depth_timestamps[1:] - depth_timestamps[:-1])
                plt.xlim([first_time, last_time])
                plt.ylim([0, 3.0/60.0])
                plt.ylabel('{}'.format(camera))
            elif camera == 'right_camera':
                plt.subplot(4,1,3)
                plt.plot(depth_timestamps[:-1], depth_timestamps[1:] - depth_timestamps[:-1])
                plt.xlim([first_time, last_time])
                plt.ylim([0, 3.0/60.0])
                plt.ylabel('{}'.format(camera))
            elif camera == 'samsung_mono':
                plt.subplot(4,1,4)
                plt.plot(depth_timestamps[:-1], depth_timestamps[1:] - depth_timestamps[:-1])
                plt.xlim([first_time, last_time])
                plt.ylim([0, 3.0/60.0])
                plt.ylabel('{}'.format(camera))
    
    if plot:
        plt.subplot(4,1,1)
        plt.title('{} {} {} depth GT frame time delta (s)'.format(category, purpose, name))

        plt.subplot(4,1,4)
        plt.xlabel('Sequence time (s)')

        plt.tight_layout()

        plt.savefig(os.path.join(plot_output_dir, '{:06d}_{}_{}_{}.png'.format(
            id, category, purpose, name)))
        plt.close()

    flea3_7      = 'flea3_7'      if 'flea3_7'      in timestamp_infos else ''
    left_camera  = 'left_camera'  if 'left_camera'  in timestamp_infos else ''
    right_camera = 'right_camera' if 'right_camera' in timestamp_infos else ''
    samsung_mono = 'samsung_mono' if 'samsung_mono' in timestamp_infos else ''

    flea3_7_data_start      = timestamp_infos['flea3_7']     ['depth_timestamps']           [0] if 'flea3_7'      in timestamp_infos else ''
    left_camera_data_start  = timestamp_infos['left_camera'] ['event_timestamps_first_last'][0] if 'left_camera'  in timestamp_infos else ''
    right_camera_data_start = timestamp_infos['right_camera']['event_timestamps_first_last'][0] if 'right_camera' in timestamp_infos else ''
    samsung_mono_data_start = timestamp_infos['samsung_mono']['event_timestamps_first_last'][0] if 'samsung_mono' in timestamp_infos else ''

    flea3_7_data_end        = timestamp_infos['flea3_7']     ['depth_timestamps']           [-1] if 'flea3_7'      in timestamp_infos else ''
    left_camera_data_end    = timestamp_infos['left_camera'] ['event_timestamps_first_last'][-1] if 'left_camera'  in timestamp_infos else ''
    right_camera_data_end   = timestamp_infos['right_camera']['event_timestamps_first_last'][-1] if 'right_camera' in timestamp_infos else ''
    samsung_mono_data_end   = timestamp_infos['samsung_mono']['event_timestamps_first_last'][-1] if 'samsung_mono' in timestamp_infos else ''

    flea3_7_data_range = flea3_7_data_end      - flea3_7_data_start      if 'flea3_7'      in timestamp_infos else ''
    left_camera_range  = left_camera_data_end  - left_camera_data_start  if 'left_camera'  in timestamp_infos else ''
    right_camera_range = right_camera_data_end - right_camera_data_start if 'right_camera' in timestamp_infos else ''
    depth_camera_range = samsung_mono_data_end - samsung_mono_data_start if 'samsung_mono' in timestamp_infos else ''

    (flea3_7_num_invalid, flea3_7_total_valid, flea3_7_max_invalid) = (
        calc_depth_valid_stats(timestamp_infos['flea3_7']['depth_timestamps'], (flea3_7_data_end, flea3_7_data_start), 2.5/30.0)      if 'flea3_7'      in timestamp_infos else ('', '', ''))
    (left_camera_num_invalid, left_camera_total_valid, left_camera_max_invalid) = (
        calc_depth_valid_stats(timestamp_infos['left_camera']['depth_timestamps'], (left_camera_data_start, left_camera_data_end), 2.5/60.0)  if 'left_camera'  in timestamp_infos else ('', '', ''))
    (right_camera_num_invalid, right_camera_total_valid, right_camera_max_invalid) = (
        calc_depth_valid_stats(timestamp_infos['right_camera']['depth_timestamps'], (right_camera_data_start, right_camera_data_end), 2.5/60.0) if 'right_camera' in timestamp_infos else ('', '', ''))
    (samsung_mono_num_invalid, samsung_mono_total_valid, samsung_mono_max_invalid) = (
        calc_depth_valid_stats(timestamp_infos['samsung_mono']['depth_timestamps'], (samsung_mono_data_start, samsung_mono_data_end), 2.5/60.0) if 'samsung_mono' in timestamp_infos else ('', '', ''))

    csv_line = [category, purpose, name,
                flea3_7, left_camera, right_camera, samsung_mono,
                flea3_7_data_range, left_camera_range, right_camera_range, depth_camera_range,
                flea3_7_total_valid, left_camera_total_valid, right_camera_total_valid, samsung_mono_total_valid,
                flea3_7_max_invalid, left_camera_max_invalid, right_camera_max_invalid, samsung_mono_max_invalid,
                flea3_7_num_invalid, left_camera_num_invalid, right_camera_num_invalid, samsung_mono_num_invalid]
    return csv_line

def group_files_by_sequence_name(files):
    files_grouped = {}

    for file in files:
        sequence_name = get_sequence_name(file)
        camera_name = get_camera_name(file)
        category = get_category(file)
        purpose = get_purpose(file)

        if sequence_name in files_grouped:
            files_grouped[sequence_name][3][camera_name] = file
        else:
            files_grouped[sequence_name] = [category, purpose, sequence_name, {camera_name: file}]

    return list(files_grouped.values())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('idir',
                        type=str)
    parser.add_argument('odir',
                        type=str)
    args = parser.parse_args()

    plot_output_dir = args.odir
    make_dir_if_needed(plot_output_dir)

    input_dir = args.idir

    file_glob = input_dir + '/*/*/*/*.npz'
    files = sorted(list(glob.glob(file_glob)))

    files_grouped_by_sequence = group_files_by_sequence_name(files)

    files_grouped_by_sequence.sort(key = lambda x: x[2])
    files_grouped_by_sequence.sort(key = lambda x: x[1])
    files_grouped_by_sequence.sort(key = lambda x: x[0])

    def analyze_timetamps_enumerate(i, args):
        return analyze_timetamps(*args, id=i)

    with Pool(int(multiprocessing.cpu_count()/2)) as p:
        csv_lines = list(tqdm(p.istarmap(analyze_timetamps_enumerate, enumerate(files_grouped_by_sequence)), total=len(files_grouped_by_sequence)))

    csv_header = (
        'category,' +
        'purpose,' +
        'sequence name,' +
        'flea3_7 (f37),' + 'left_camera (le),' + 'right_camera (re),' + 'samsung_mono (se),' +
        'f37 sec data,' + 'le sec data,' + 'lr sec data,' + 'se sec data,' +
        'f37 sec depth,' + 'le sec depth,' + 'lr sec depth,' + 'se sec depth,' +
        'f37 max invalid,' + 'le max_invalid,' + 'lr max_invalid,' + 'se max_invalid,' +
        'f37 num invalid,' + 'le num invalid,' + 'lr num invalid,' + 'se num invalid'
    )

    csv_file = csv_header + '\n'

    for line in csv_lines:
        txt_csv_line = ''
        for l in line:
            txt_csv_line = txt_csv_line + '{},'.format(l)
        txt_csv_line = txt_csv_line + '\n'

        csv_file = csv_file + txt_csv_line

    with open('sequences_analyzed.csv', 'w') as file:
        file.write(csv_file)
