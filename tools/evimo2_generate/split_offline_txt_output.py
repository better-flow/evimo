#!/usr/bin/python3

import argparse
import multiprocessing
from multiprocessing import Pool
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os, sys, math, signal, glob
import cv2
import pydvs
from tqdm import tqdm
import shutil
import pprint

# aos2soa does not suffice because there are dropped poses
def frames_meta_to_arrays(all_objects_pose_list):
    objects_arrays = {}

    for objects_pose in all_objects_pose_list:
        for obj_id in objects_pose:
            if obj_id == 'ts':
                if 'ts' not in objects_arrays:
                    objects_arrays['ts'] = []
                objects_arrays['ts'].append(objects_pose['ts'])

            if obj_id == 'cam' or obj_id.isnumeric():
                if obj_id not in objects_arrays:
                    objects_arrays[obj_id] = {}
                    objects_arrays[obj_id]['ts'] = []
                    objects_arrays[obj_id]['pos'] = {}
                    objects_arrays[obj_id]['pos']['t'] = {}
                    objects_arrays[obj_id]['pos']['t']['x'] = []
                    objects_arrays[obj_id]['pos']['t']['y'] = []
                    objects_arrays[obj_id]['pos']['t']['z'] = []
                    objects_arrays[obj_id]['pos']['rpy'] = {}
                    objects_arrays[obj_id]['pos']['rpy']['r'] = []
                    objects_arrays[obj_id]['pos']['rpy']['p'] = []
                    objects_arrays[obj_id]['pos']['rpy']['y'] = []
                    objects_arrays[obj_id]['pos']['q'] = {}
                    objects_arrays[obj_id]['pos']['q']['w'] = []
                    objects_arrays[obj_id]['pos']['q']['x'] = []
                    objects_arrays[obj_id]['pos']['q']['y'] = []
                    objects_arrays[obj_id]['pos']['q']['z'] = []

                objects_arrays[obj_id]['ts'].append(objects_pose[obj_id]['ts'])
                objects_arrays[obj_id]['pos']['t']['x'].append(objects_pose[obj_id]['pos']['t']['x'])
                objects_arrays[obj_id]['pos']['t']['y'].append(objects_pose[obj_id]['pos']['t']['y'])
                objects_arrays[obj_id]['pos']['t']['z'].append(objects_pose[obj_id]['pos']['t']['z'])
                objects_arrays[obj_id]['pos']['rpy']['r'].append(objects_pose[obj_id]['pos']['rpy']['r'])
                objects_arrays[obj_id]['pos']['rpy']['p'].append(objects_pose[obj_id]['pos']['rpy']['p'])
                objects_arrays[obj_id]['pos']['rpy']['y'].append(objects_pose[obj_id]['pos']['rpy']['y'])
                objects_arrays[obj_id]['pos']['q']['w'].append(objects_pose[obj_id]['pos']['q']['w'])
                objects_arrays[obj_id]['pos']['q']['x'].append(objects_pose[obj_id]['pos']['q']['x'])
                objects_arrays[obj_id]['pos']['q']['y'].append(objects_pose[obj_id]['pos']['q']['y'])
                objects_arrays[obj_id]['pos']['q']['z'].append(objects_pose[obj_id]['pos']['q']['z'])
    return objects_arrays


def save_plot_frame_pacing(camera_full_trajectory, camera_gt_frame_times, file_name):
    plt.rcParams['lines.linewidth'] = 0.8
    fig, axs = plt.subplots(4, 1)

    for camera in camera_full_trajectory:
        full_trajectory = camera_full_trajectory[camera]
        full_trajectory_plottable = frames_meta_to_arrays(full_trajectory)
        gt_frame_times = camera_gt_frame_times[camera]

        gt_frame_time_diffs = np.diff(gt_frame_times)

        if camera == 'flea3_7':
            cam_plot_id = 0
        elif camera == 'left_camera':
            cam_plot_id = 1
        elif camera == 'right_camera':
            cam_plot_id = 2
        elif camera == 'samsung_mono':
            cam_plot_id = 3

        axs[cam_plot_id].plot(gt_frame_times[:-1], gt_frame_time_diffs, label=camera)
        axs[cam_plot_id].set_xlim([full_trajectory_plottable['ts'][0], full_trajectory_plottable['ts'][-1]])
        axs[cam_plot_id].set_ylim([0, 3.0/60.0])
        axs[cam_plot_id].grid()

    axs[0].set_ylabel('flea3_7')
    axs[1].set_ylabel('left_camera')
    axs[2].set_ylabel('right_camera')
    axs[3].set_ylabel('samsung_mono')

    axs[0].set_title('GT frame pacing (delta seconds)')
    axs[3].set_xlabel('time (s)')

    plt.tight_layout()
    plt.savefig(file_name, dpi=400, bbox_inches='tight')
    #plt.show()

# Determine sets of gt_timestamps that meet certain criteria about gaps
def determine_gt_splits(gt_timestamps, min_interval_sec, max_gap_s):
    gt_timestamps_split = [[gt_timestamps[0],]]

    # Split based on maximum gap
    for t0, t1 in zip(gt_timestamps[:-1], gt_timestamps[1:]):
        if t1 - t0 > max_gap_s:
            gt_timestamps_split.append([t1])
        else:
            gt_timestamps_split[-1].append(t1)

    # Remove sections with length less than min_interval_sec
    gt_timestamps_split_long_enough = []
    for timestamps in gt_timestamps_split:
        if timestamps[-1] - timestamps[0] >= min_interval_sec:
            gt_timestamps_split_long_enough.append(timestamps)

    return gt_timestamps_split_long_enough

def slice_txt_dataset(input_folder, output_folder, gt_times, dataset_txt):
    pydvs.replace_dir(output_folder)

    # meta.txt, gt_frames/classical_frames
    sliced_dataset_txt = {}
    for key in dataset_txt:
        if key == 'meta':
            sliced_dataset_txt[key] = dataset_txt[key]

        elif key == 'frames':
            if key not in sliced_dataset_txt:
                sliced_dataset_txt[key] = []

            for frame in dataset_txt[key]:
                if frame['ts'] >= gt_times[0] and frame['ts'] <= gt_times[-1]:
                    sliced_dataset_txt[key].append(frame)
                    if 'gt_frame' in frame:
                        input_frame_name = os.path.join(input_folder, frame['gt_frame'])
                        output_frame_name = os.path.join(output_folder, frame['gt_frame'])
                        shutil.copyfile(input_frame_name, output_frame_name) # Replace with move
                    if 'classical_frame' in frame:
                        input_frame_name = os.path.join(input_folder, frame['classical_frame'])
                        output_frame_name = os.path.join(output_folder, frame['classical_frame'])
                        shutil.copyfile(input_frame_name, output_frame_name) # Replace with move

        elif key == 'full_trajectory':
            if key not in sliced_dataset_txt:
                sliced_dataset_txt[key] = []

            for traj in dataset_txt[key]:
                if traj['ts'] >= gt_times[0] and traj['ts'] <= gt_times[-1]:
                    sliced_dataset_txt[key].append(traj)

        elif key == 'imu':
            if key not in sliced_dataset_txt:
                sliced_dataset_txt[key] = {}

            for imu_source in dataset_txt[key]:
                if imu_source not in sliced_dataset_txt[key]:
                    sliced_dataset_txt[key][imu_source] = []

                for imu_measurement in dataset_txt[key][imu_source]:
                    if imu_measurement['ts'] >= gt_times[0] and imu_measurement['ts'] <= gt_times[-1]:
                        sliced_dataset_txt[key][imu_source].append(imu_measurement)

        else:
            raise Exception('Unknown key, cannot slice')

    output_meta_txt = os.path.join(output_folder, 'meta.txt')
    with open(output_meta_txt, 'w') as f:
        f.write(pprint.pformat(sliced_dataset_txt, compact=True))

    # sliced_dataset_txt_read_back = eval(open(output_meta_txt).read())

    # with open(output_meta_txt+'.bak', 'w') as f:
    #     f.write(pprint.pformat(sliced_dataset_txt_read_back, compact=True))

    events, _ = pydvs.read_event_file_txt(os.path.join(input_folder, 'events.txt'), 0.01)

    if events.shape[0] > 0:
        left_index  = np.searchsorted(events[:, 0], gt_times[0])
        right_index = np.searchsorted(events[:, 0], gt_times[-1], side='right')

        # events_sliced = events[left_index:right_index, :]
        # print(events_sliced)

        # https://stackoverflow.com/questions/83329/how-can-i-extract-a-predetermined-range-of-lines-from-a-text-file-on-unix
        # sed -n '16224,16482p;16483q' filename > newfile
        sed_string = '{},{}p;{}q'.format(left_index+1, right_index, right_index+1)
        sed_process = subprocess.Popen('sed -n ' + "'" + sed_string + "' " + os.path.join(input_folder, 'events.txt') + ' > ' + os.path.join(output_folder, 'events.txt'), shell=True)
        sed_process.wait()

        #print(events_sliced)
    else:
        with open(os.path.join(output_folder, 'events.txt'), 'w') as f:
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        type=str,
                        default='.',
                        required=False)
    args = parser.parse_args()

    print (pydvs.okb("Opening"), args.base_dir)

    event_cameras_list = ['samsung_mono', 'left_camera', 'right_camera']
    cameras_list = ['flea3_7',] + event_cameras_list
    camera_dataset_txt = {}
    camera_gt_frame_times = {}
    camera_full_trajectory = {}

    for camera in cameras_list:
        meta_txt_file = os.path.join(args.base_dir, camera, 'ground_truth', 'meta.txt')

        if os.path.exists(meta_txt_file):
            dataset_txt = eval(open(meta_txt_file).read())
            camera_dataset_txt[camera] = dataset_txt

            NUM_FRAMES = len(dataset_txt['frames'])
            frames_meta = dataset_txt['frames']

            NUM_GT_FRAMES = 0
            for frame in frames_meta:
                if 'gt_frame' in frame:
                    NUM_GT_FRAMES+=1

            print (pydvs.okb(camera))
            print (pydvs.okb("Frames:"), NUM_FRAMES)
            print (pydvs.okb("Frames with GT:"), NUM_GT_FRAMES)

            frames_meta = camera_dataset_txt[camera]['frames']
            gt_frame_times = []
            for frame in frames_meta:
                if 'gt_frame' in frame:
                    gt_frame_times.append(frame['ts'])
            gt_frame_times = np.array(gt_frame_times)

            camera_gt_frame_times[camera] = gt_frame_times;
            camera_full_trajectory[camera] = dataset_txt['full_trajectory']

    # Create a plot
    save_plot_frame_pacing(camera_full_trajectory, camera_gt_frame_times, os.path.join(args.base_dir, 'frame_time_plot.pdf'))

    # Check that frame times are consistent between event cameras
    to_compare_gt_frame_times = None
    for camera in event_cameras_list:
        if to_compare_gt_frame_times is None:
            to_compare_gt_frame_times = (camera, camera_gt_frame_times[camera])
        else:
            if not np.all(to_compare_gt_frame_times[1] == camera_gt_frame_times[camera]):
                print(to_compare_gt_frame_times[1])
                print(camera_gt_frame_times[camera])
                raise Exception('event camera frame times are not consistent between cameras {} and {}'.format(to_compare_gt_frame_times[0], camera))


    # Because all events cameras have identical GT times and there is always an event camera
    # we can use just one event camera's GT timestamps to determine the splits
    # Use the one that everything was compared to
    gt_timestamps_split = determine_gt_splits(to_compare_gt_frame_times[1], min_interval_sec=0.4, max_gap_s=1.0)

    for i, gt_times in enumerate(gt_timestamps_split):
        print('Split {}'.format(i))
        print(gt_times)

    for i, gt_times in enumerate(gt_timestamps_split):
        for camera in cameras_list:
            if camera in camera_gt_frame_times:
                input_folder = os.path.join(args.base_dir, camera, 'ground_truth')
                output_folder = os.path.join(args.base_dir, camera, 'ground_truth_{:06d}'.format(i))
                slice_txt_dataset(input_folder, output_folder, gt_times, camera_dataset_txt[camera])

    print (pydvs.okg("\nDone.\n"))
