#!/usr/bin/python3
import argparse
import cv2
import numpy as np
import glob
import pickle
import os
import argparse

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from extract_event_and_frame_times import (get_sequence_name,
                                           get_camera_name,
                                           get_category,
                                           get_purpose,
                                           make_dir_if_needed)

IMG_HEIGHT = int(240*1.25)
IMG_WIDTH = int(320*1.25)

MAX_VIS_DEPTH = 1.5
EVENT_COUNT_DT = 0.005
EVENTS_IN_EVENT_COUNT = 5000


# From pydvs evimo-gen.py
def mask_to_color(mask):
    colors = [[84, 71, 140],   [44, 105, 154],  [4, 139, 168],
              [13, 179, 158],  [22, 219, 147],  [131, 227, 119],
              [185, 231, 105], [239, 234, 90],  [241, 196, 83],
              [242, 158, 76],  [239, 71, 111],  [255, 209, 102],
              [6, 214, 160],   [17, 138, 178],  [7, 59, 76],
              [6, 123, 194],   [132, 188, 218], [236, 195, 11],
              [243, 119, 72],  [213, 96, 98]]

    cmb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    m_ = np.max(mask) + 500
    m_ = max(m_, 3500)

    maxoid = int(m_ / 1000)
    for i in range(maxoid):
        cutoff_lo = 1000.0 * (i + 1.0) - 5
        cutoff_hi = 1000.0 * (i + 1.0) + 5
        cmb[np.where(np.logical_and(mask>=cutoff_lo, mask<=cutoff_hi))] = np.array(colors[i % len(colors)])
    return cmb


# For some reason this is faster than numpy's search sorted
# by about 10000x (not an exaggeration)
def my_search_sorted(array, t):
    start = 0
    end = array.shape[0] - 1

    while start <= end:
        mid = int((end + start) / 2)
        i = array[mid]
        if i <= t:
            start = mid + 1
        elif i > t:
            end = mid - 1

    return mid

def count_events_polarity(events, out_shape):
    event_count = np.zeros((out_shape), dtype=np.float32)
    event_count[events[:, 1], events[:, 0]] += (2*events[:, 2].astype(np.float32) - 1)
    return event_count

def get_frame_by_index(frames, index):
    frames_names = list(frames.keys())
    frames_names.sort()
    frame_name = frames_names[index]
    return np.copy(frames[frame_name]) # To extract and keep in RAM

def visualize_event_camera(t, events, depth, depth_timestamps, camera_resolution, out_width, out_height):
    # Make images for left_camera if in sequence else black
    if events is not None:
        # Black if there is no data at this time
        if t < events[0, 0] or t > events[-1, 0] + 1/60.0:
            event_count = None
            event_count_normalized = None
        # There is data, make visualization
        else:
            i_left = my_search_sorted(events[:, 0], t) - 1
            i_right = my_search_sorted(events[:, 0], t + EVENT_COUNT_DT) - 1
            #i_right = i_left + EVENTS_IN_EVENT_COUNT

            events = events[i_left:i_right, 1:].astype(np.uint32)

            event_count = count_events_polarity(events, camera_resolution)
            event_count = cv2.resize(event_count, dsize=(out_width, out_height))

            max_count = np.max(np.abs(event_count))
            event_count_normalized = (event_count * 127 / max_count).astype(np.int8)

        # Visualize depth if possible
        if not (t < depth_timestamps[0] or  t > depth_timestamps[-1] + 1/60.0):
            i = np.searchsorted(depth_timestamps, t, side='right') - 1
            d = get_frame_by_index(depth, i)
            d = cv2.resize(d, dsize=(out_width, out_height))
            d = np.clip(d.astype(np.float32) * (255 / MAX_VIS_DEPTH / 1000), 0.0, 255.0).astype(np.uint8)
            depth_bgr = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
        else:
            depth_bgr = None

        # Compose images as appropriate
        if event_count_normalized is None and depth_bgr is None:
            event_bgr = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        elif event_count_normalized is None:
            event_bgr = depth_bgr
        elif depth_bgr is None:
            event_bgr = np.zeros((out_height, out_width, 3), dtype=np.uint8)
            pos = event_count_normalized >= 0
            event_bgr[pos, 1] = event_count_normalized[pos]
            neg = ~pos
            event_bgr[neg, 2] = -event_count_normalized[neg]
        else:
            # event_count_bgr = np.dstack((np.zeros((out_height, out_width), dtype=np.uint8), event_count_normalized, np.zeros((out_height, out_width), dtype=np.uint8)))
            event_count_bgr = np.zeros((out_height, out_width, 3), dtype=np.uint8)
            pos = event_count_normalized >= 0
            event_count_bgr[pos, 1] = event_count_normalized[pos]
            neg = ~pos
            event_count_bgr[neg, 2] = -event_count_normalized[neg]

            event_bgr_float = depth_bgr.astype(np.float32) + event_count_bgr.astype(np.float32)
            event_bgr = np.clip(event_bgr_float, 0, 255).astype(np.uint8)
    else:
        event_count_normalized = None
        event_bgr = np.zeros((out_height, out_width, 3), dtype=np.uint8)

    # The codes depending on this function expect all elements greater than 0
    if event_count_normalized is not None:
        event_count_normalized = np.abs(event_count_normalized)

    return event_bgr, event_count_normalized

def visualize_event_camera_mask(t, timestamps, masks, out_width, out_height, event_count_normalized=None):
    if masks is not None:
        i = np.searchsorted(timestamps, t, side='right') - 1
        m = get_frame_by_index(masks, i)
        m = cv2.resize(m, dsize=(out_width, out_height), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        col_mask = mask_to_color(m)
        col_mask = col_mask.astype(np.uint8)
    else:
        col_mask = np.zeros((out_height, out_width, 3), dtype=np.uint8)

    event_count_bgr = np.dstack((event_count_normalized, event_count_normalized, event_count_normalized))

    col_mask = col_mask.astype(np.float32) + event_count_bgr.astype(np.float32)
    col_mask = np.clip(col_mask, 0, 255).astype(np.uint8)

    return col_mask

def on_trackbar(t_ms):
    t = (t_ms / 1000.0) + t_start

    # Make images for flea3_7 if in sequence else black
    if flea3_7_classical_timestamps is not None:
        # Black if there is no data at this time
        if t < flea3_7_classical_timestamps[0] or  t > flea3_7_classical_timestamps[-1] + 1/30.0:
            flea3_7_img_bgr = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            flea3_7_depth_bgr = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        # There is data, make visualization frames
        else:
            # Visualize color image with mask on top
            i_classical = np.searchsorted(flea3_7_classical_timestamps, t) - 1
            flea3_7_img = get_frame_by_index(flea3_7_data, i_classical)
            flea3_7_img_bgr = cv2.resize(flea3_7_img, dsize=(IMG_WIDTH, IMG_HEIGHT))

            i_depth = np.searchsorted(flea3_7_depth_timestamps, t) - 1
            flea3_7_m = get_frame_by_index(flea3_7_mask, i_depth)
            flea3_7_m = cv2.resize(flea3_7_m, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            col_mask = mask_to_color(flea3_7_m)

            flea3_7_m = (255 * (flea3_7_m - np.nanmin(flea3_7_m)) / (np.nanmax(flea3_7_m) - np.nanmin(flea3_7_m))).astype(np.uint8)
            mask_more_than_0 = flea3_7_m > 0

            flea3_7_img_bgr[mask_more_than_0] = flea3_7_img_bgr[mask_more_than_0] * 0.2 + col_mask[mask_more_than_0] * 0.8

            # Visualize depth
            flea3_7_d = get_frame_by_index(flea3_7_depth, i_depth)
            flea3_7_d = cv2.resize(flea3_7_d, dsize=(IMG_WIDTH, IMG_HEIGHT))
            flea3_7_d = np.clip(flea3_7_d.astype(np.float32) * (255 / MAX_VIS_DEPTH / 1000), 0.0, 255.0).astype(np.uint8)
            flea3_7_depth_bgr = cv2.cvtColor(flea3_7_d, cv2.COLOR_GRAY2BGR)
    else:
        flea3_7_img_bgr = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        flea3_7_depth_bgr = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

    left_camera_event_bgr, left_camera_event_count = visualize_event_camera(t, left_camera_events, left_camera_depth, left_camera_timestamps, left_camera_resolution, IMG_WIDTH, IMG_HEIGHT)
    right_camera_event_bgr, right_camera_event_count = visualize_event_camera(t, right_camera_events, right_camera_depth, right_camera_timestamps, right_camera_resolution, IMG_WIDTH, IMG_HEIGHT)
    samsung_mono_event_bgr, samsung_mono_event_count = visualize_event_camera(t, samsung_mono_events, samsung_mono_depth, samsung_mono_timestamps, samsung_mono_resolution, IMG_WIDTH, IMG_HEIGHT)

    left_camera_mask_bgr  = visualize_event_camera_mask(t, left_camera_timestamps, left_camera_mask, IMG_WIDTH, IMG_HEIGHT, left_camera_event_count)
    right_camera_mask_bgr = visualize_event_camera_mask(t, right_camera_timestamps, right_camera_mask, IMG_WIDTH, IMG_HEIGHT, right_camera_event_count)
    samsung_mono_mask_bgr = visualize_event_camera_mask(t, samsung_mono_timestamps, samsung_mono_mask, IMG_WIDTH, IMG_HEIGHT, samsung_mono_event_count)

    red_line_loc = (t - t_start) * pixels_per_second + first_tick
    plot_bgr_line = cv2.line(np.copy(plot_bgr), (int(red_line_loc), 0),(int(red_line_loc), plot_bgr.shape[0]), (0, 0, 255), 1)

    top_row = np.hstack((cv2.flip(flea3_7_depth_bgr, -1), samsung_mono_event_bgr, cv2.flip(left_camera_event_bgr, -1), right_camera_event_bgr))
    mid_row = np.hstack((cv2.flip(flea3_7_img_bgr, -1), samsung_mono_mask_bgr, cv2.flip(left_camera_mask_bgr, -1), right_camera_mask_bgr))
    zeros = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    bottom_row = np.hstack((plot_bgr_line, zeros, zeros, zeros))


    total_image = np.vstack((top_row, mid_row, bottom_row))
    cv2.imshow(title_window, total_image)


def snap_to_time(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if y >= 2*IMG_HEIGHT and y < 3 * IMG_HEIGHT:
            x = x - IMG_WIDTH * 0
            last_tick = (t_end - t_start) * pixels_per_second + first_tick
            if x >= first_tick and x <= last_tick:
                t = ((t_end - t_start) * (x - first_tick) / (last_tick - first_tick)) + t_start
                t_id = int((t-t_start) * 1000)
                cv2.setTrackbarPos(trackbar_name, title_window, t_id)

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

    return files_grouped


def make_timestamp_plot(flea3_7_timestamps, left_camera_timestamps, right_camera_timestamps, samsung_mono_timestamps):
    fig = plt.figure()
    canvas = FigureCanvas(fig)

    def get_first_tick_and_pixels_per_second(ax):
        # https://stackoverflow.com/a/44012582
        # https://discourse.matplotlib.org/t/pixel-position-of-x-y-axis/11098/5
        xtickslocs = ax.get_xticks()
        ymin, _ = ax.get_ylim()
        xticks_pixels = ax.transData.transform([(xtick, ymin) for xtick in xtickslocs])
        first_tick = xticks_pixels[0][0]
        pixels_per_second = (xticks_pixels[1][0] - xticks_pixels[0][0]) / (xtickslocs[1] - xtickslocs[0])
        axis_loc = ax.bbox.extents
        first_tick = axis_loc[0]

        return first_tick, pixels_per_second

    if flea3_7_timestamps is not None:
        ax = plt.subplot(4,1,1)
        plt.plot(flea3_7_timestamps[:-1], flea3_7_timestamps[1:] - flea3_7_timestamps[:-1])
        plt.xlim([t_start, t_end])
        plt.ylim([0, 3.0/60.0])
        plt.ylabel('{}'.format('flea3_7'))

    if left_camera_timestamps is not None:
        ax = plt.subplot(4,1,2)
        plt.plot(left_camera_timestamps[:-1], left_camera_timestamps[1:] - left_camera_timestamps[:-1])
        plt.xlim([t_start, t_end])
        plt.ylim([0, 3.0/60.0])
        plt.ylabel('{}'.format('left_camera'))

    if right_camera_timestamps is not None:
        ax = plt.subplot(4,1,3)
        plt.plot(right_camera_timestamps[:-1], right_camera_timestamps[1:] - right_camera_timestamps[:-1])
        plt.xlim([t_start, t_end])
        plt.ylim([0, 3.0/60.0])
        plt.ylabel('{}'.format('right_camera'))

    if samsung_mono_timestamps is not None:
        ax = plt.subplot(4,1,4)
        plt.plot(samsung_mono_timestamps[:-1], samsung_mono_timestamps[1:] - samsung_mono_timestamps[:-1])
        plt.xlim([t_start, t_end])
        plt.ylim([0, 3.0/60.0])
        plt.ylabel('{}'.format('samsung_mono'))

    plt.subplot(4,1,1)
    plt.title('depth GT frame time delta (s)')
    plt.subplot(4,1,4)
    plt.xlim([t_start, t_end])
    plt.xlabel('Sequence time (s)')

    plt.tight_layout()

    canvas.draw()

    for i, timestamps in enumerate((flea3_7_timestamps, left_camera_timestamps, right_camera_timestamps, samsung_mono_timestamps)):
        if type(timestamps) == np.ndarray:
            ax = plt.subplot(4,1,i+1)
            first_tick, pixels_per_second = get_first_tick_and_pixels_per_second(ax)
            break

    width, height = fig.get_size_inches() * fig.get_dpi()
    data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.close()
    plot_bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

    return plot_bgr, first_tick, pixels_per_second

# Load an EVIMO2_v2 npz format list of events into RAM
def load_events(folder):
    events_t  = np.load(os.path.join(folder, 'dataset_events_t.npy'), mmap_mode='r')
    events_xy = np.load(os.path.join(folder, 'dataset_events_xy.npy'), mmap_mode='r')
    events_p  = np.load(os.path.join(folder, 'dataset_events_p.npy'), mmap_mode='r')

    events_t = np.atleast_2d(events_t.astype(np.float32)).transpose()
    events_p = np.atleast_2d(events_p.astype(np.float32)).transpose()

    events = np.hstack((events_t,
                        events_xy.astype(np.float32),
                        events_p))
    return events


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View all cameras of a sequence with GT depth overlaid to see availability')
    parser = argparse.ArgumentParser()
    parser.add_argument('--idir', type=str, help='Directory containing npz file tree')
    parser.add_argument('--seq', default='scene13_dyn_test_00', help='Sequence name')
    args = parser.parse_args()

    data_base_folder = args.idir
    file_glob = data_base_folder + '/*/*/*/*'
    files = sorted(list(glob.glob(file_glob)))

    files_grouped_by_sequence = group_files_by_sequence_name(files)
    folders = files_grouped_by_sequence[args.seq][3]

    print('Opening npy files')
    if 'flea3_7' in folders:
        print('flea3_7 frames')
        flea3_7_data = np.load(os.path.join(folders['flea3_7'], 'dataset_classical.npz'))
        flea3_7_depth = np.load(os.path.join(folders['flea3_7'], 'dataset_depth.npz'))
        flea3_7_mask = np.load(os.path.join(folders['flea3_7'], 'dataset_mask.npz'))
        print('flea3_7_dataset_info')
        meta = np.load(os.path.join(folders['flea3_7'], 'dataset_info.npz'), allow_pickle=True)['meta'].item()
        frame_infos = meta['frames']
        flea3_7_depth_timestamps = []
        flea3_7_classical_timestamps = []
        for frame in frame_infos:
            if 'gt_frame' in frame:
                flea3_7_depth_timestamps.append(frame['ts'])
            if 'classical_frame' in frame:
                flea3_7_classical_timestamps.append(frame['ts'])
        flea3_7_depth_timestamps = np.array(flea3_7_depth_timestamps)
        flea3_7_classical_timestamps = np.array(flea3_7_classical_timestamps)
    else:
        flea3_7_data = None
        flea3_7_depth = None
        flea3_7_mask = None
        flea3_7_depth_timestamps = None
        flea3_7_classical_timestamps = None

    if 'left_camera' in folders:
        print('left_camera frames')
        left_camera_events = load_events(folders['left_camera'])
        left_camera_depth = np.load(os.path.join(folders['left_camera'], 'dataset_depth.npz'))
        left_camera_mask = np.load(os.path.join(folders['left_camera'], 'dataset_mask.npz'))
        print('left_camera dataset_info')
        meta = np.load(os.path.join(folders['left_camera'], 'dataset_info.npz'), allow_pickle=True)['meta'].item()
        frame_infos = meta['frames']
        left_camera_timestamps = []
        for frame in frame_infos:
            if 'gt_frame' in frame:
                left_camera_timestamps.append(frame['ts'])
        left_camera_timestamps = np.array(left_camera_timestamps)
    else:
        left_camera_events = None
        left_camera_mask = None
        left_camera_depth = None
        left_camera_timestamps = None

    if 'right_camera' in folders:
        print('right_camera frames')
        right_camera_events = load_events(folders['right_camera'])
        right_camera_depth = np.load(os.path.join(folders['right_camera'], 'dataset_depth.npz'))
        right_camera_mask = np.load(os.path.join(folders['right_camera'], 'dataset_mask.npz'))
        print('right_camera dataset_info')
        meta = np.load(os.path.join(folders['right_camera'], 'dataset_info.npz'), allow_pickle=True)['meta'].item()
        frame_infos = meta['frames']
        right_camera_timestamps = []
        for frame in frame_infos:
            if 'gt_frame' in frame:
                right_camera_timestamps.append(frame['ts'])
        right_camera_timestamps = np.array(right_camera_timestamps)
    else:
        right_camera_events = None
        right_camera_mask = None
        right_camera_depth = None
        right_camera_timestamps = None

    if 'samsung_mono' in folders:
        print('samsung_mono frames')
        samsung_mono_events = load_events(folders['samsung_mono'])
        samsung_mono_depth = np.load(os.path.join(folders['samsung_mono'], 'dataset_depth.npz'))
        samsung_mono_mask = np.load(os.path.join(folders['samsung_mono'], 'dataset_mask.npz'))
        print('samsung_mono dataset_info')
        meta = np.load(os.path.join(folders['samsung_mono'], 'dataset_info.npz'), allow_pickle=True)['meta'].item()
        frame_infos = meta['frames']
        samsung_mono_timestamps = []
        for frame in frame_infos:
            if 'gt_frame' in frame:
                samsung_mono_timestamps.append(frame['ts'])
        samsung_mono_timestamps = np.array(samsung_mono_timestamps)
    else:
        samsung_mono_events = None
        samsung_mono_mask = None
        samsung_mono_depth = None
        samsung_mono_timestamps = None

    flea3_7_resolution = (1552, 2080)
    left_camera_resolution  = (480, 640)
    right_camera_resolution = (480, 640)
    samsung_mono_resolution = (480, 640)

    # Can't just use a max because of all the None's
    t_start = None
    t_end = None
    for timestamps in (flea3_7_depth_timestamps, flea3_7_classical_timestamps, left_camera_timestamps, right_camera_timestamps, samsung_mono_timestamps,
                       left_camera_events, right_camera_events, samsung_mono_events):
        if timestamps is not None:
            if len(timestamps.shape) == 1:
                new_t_start = timestamps[0]
                new_t_end   = timestamps[-1]
            else:
                new_t_start = timestamps[0, 0]
                new_t_end   = timestamps[-1, 0]

            if t_start is None or new_t_start < t_start:
                t_start = new_t_start

            if t_end is None or new_t_end > t_end:
                t_end = new_t_end

    cv2.setNumThreads(4) # OpenCV threading is very wasteful, limit it

    print('Making plot')
    plot_bgr_orig, first_tick_orig, pixels_per_second_orig = make_timestamp_plot(flea3_7_depth_timestamps, left_camera_timestamps, right_camera_timestamps, samsung_mono_timestamps)
    plot_bgr = cv2.resize(plot_bgr_orig, dsize=(IMG_WIDTH, IMG_HEIGHT))
    first_tick = first_tick_orig * (IMG_HEIGHT / plot_bgr_orig.shape[0])
    pixels_per_second = pixels_per_second_orig  * (IMG_HEIGHT / plot_bgr_orig.shape[0])

    title_window = args.seq
    cv2.namedWindow(title_window)

    trackbar_name = 't - t_start (ms)'
    slider_max = int(1000 * (t_end - t_start))
    cv2.createTrackbar(trackbar_name, title_window , int(slider_max/2), slider_max, on_trackbar)
    cv2.setMouseCallback(title_window, snap_to_time)

    print('Visualizing first slider position')
    on_trackbar(int(slider_max/2))

    cv2.waitKey()
