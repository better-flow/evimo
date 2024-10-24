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

IMG_HEIGHT = int(240*1.5)
IMG_WIDTH = int(320*1.5)

MAX_VIS_DEPTH = 1.5
EVENT_COUNT_DT = 0.002
EVENTS_IN_EVENT_COUNT = 5000


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

def count_events(events, out_shape):
    event_count = np.zeros((out_shape), dtype=np.float32)
    event_count[events[:, 1], events[:, 0]] += 1
    return event_count

def visualize_event_camera(t, events, depth, depth_timestamps, camera_resolution, out_width, out_height):
    min_event_brightness = 100

    # Make images for left_camera if in sequence else black
    if events is not None:
        # Black if there is no data at this time
        if t < events[0, 0] or t > events[-1, 0] + 1/60.0:
            event_count_green = None
        # There is data, make visualization
        else:
            i_left = my_search_sorted(events[:, 0], t) - 1
            i_right = my_search_sorted(events[:, 0], t + EVENT_COUNT_DT) - 1
            #i_right = i_left + EVENTS_IN_EVENT_COUNT

            events = events[i_left:i_right, 1:].astype(np.uint32)

            event_count = count_events(events, camera_resolution)
            event_count = cv2.resize(event_count, dsize=(out_width, out_height))

            max_count = np.max(event_count)
            event_count_green = (event_count * ((255-min_event_brightness) / max_count)).astype(np.uint8) + min_event_brightness

        # Visualize depth if possible
        if not (t < depth_timestamps[0] or  t > depth_timestamps[-1] + 1/60.0):
            i = np.searchsorted(depth_timestamps, t, side='right') - 1
            d = depth[i, :, :]
            d = cv2.resize(d, dsize=(out_width, out_height))
            d = np.clip(d.astype(np.float32) * (255 / MAX_VIS_DEPTH / 1000), 0.0, 255.0).astype(np.uint8)
            depth_bgr = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
        else:
            depth_bgr = None

        # Compose images as appropriate
        if event_count_green is None and depth_bgr is None:
            event_bgr = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        elif event_count_green is None:
            event_bgr = depth_bgr
        elif depth_bgr is None:
            event_bgr = np.zeros((out_height, out_width, 3), dtype=np.uint8)
            event_bgr[:, :, 1][event_count_green > min_event_brightness] = event_count_green[event_count_green > min_event_brightness]
        else:
            depth_bgr[:, :, 0][event_count_green > min_event_brightness] = 0
            depth_bgr[:, :, 1][event_count_green > min_event_brightness] = event_count_green[event_count_green > min_event_brightness]
            depth_bgr[:, :, 2][event_count_green > min_event_brightness] = 0
            event_bgr = depth_bgr
    else:
        event_bgr = np.zeros((out_height, out_width, 3), dtype=np.uint8)

    return event_bgr

def on_trackbar(t_ms):
    t = t_ms / 1000.0

    # Make images for flea3_7 if in sequence else black
    if flea3_7_timestamps is not None:
        # Black if there is no data at this time
        if t < flea3_7_timestamps[0] or  t > flea3_7_timestamps[-1] + 1/30.0:
            flea3_7_img_bgr = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            flea3_7_depth_bgr = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        # There is data, make visualization frames
        else:
            i = my_search_sorted(flea3_7_timestamps, t) - 1
            flea3_7_img = flea3_7_data[i, :, :, :]
            flea3_7_img_bgr = cv2.resize(flea3_7_img, dsize=(IMG_WIDTH, IMG_HEIGHT))

            flea3_7_d = flea3_7_depth[i, :, :]
            flea3_7_d = cv2.resize(flea3_7_d, dsize=(IMG_WIDTH, IMG_HEIGHT))
            flea3_7_d = np.clip(flea3_7_d.astype(np.float32) * (255 / MAX_VIS_DEPTH / 1000), 0.0, 255.0).astype(np.uint8)
            flea3_7_depth_bgr = cv2.cvtColor(flea3_7_d, cv2.COLOR_GRAY2BGR)

            flea3_7_img_bgr = cv2.addWeighted(flea3_7_img_bgr, 0.5, flea3_7_depth_bgr, 0.5, 0.0)

    else:
        flea3_7_img_bgr = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        flea3_7_depth_bgr = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

    left_camera_event_bgr = visualize_event_camera(t, left_camera_events, left_camera_depth, left_camera_timestamps, left_camera_resolution, IMG_WIDTH, IMG_HEIGHT)
    right_camera_event_bgr = visualize_event_camera(t, right_camera_events, right_camera_depth, right_camera_timestamps, right_camera_resolution, IMG_WIDTH, IMG_HEIGHT)
    samsung_mono_event_bgr = visualize_event_camera(t, samsung_mono_events, samsung_mono_depth, samsung_mono_timestamps, samsung_mono_resolution, IMG_WIDTH, IMG_HEIGHT)

    red_line_loc = (t - t_start) * pixels_per_second + first_tick
    plot_bgr_line = cv2.line(np.copy(plot_bgr), (int(red_line_loc), 0),(int(red_line_loc), plot_bgr.shape[0]), (0, 0, 255), 1)

    top_row = np.hstack((cv2.flip(flea3_7_img_bgr, -1), cv2.flip(flea3_7_depth_bgr, -1), samsung_mono_event_bgr))
    bottom_row = np.hstack((cv2.flip(left_camera_event_bgr, -1), right_camera_event_bgr, plot_bgr_line))

    total_image = np.vstack((top_row, bottom_row))
    cv2.imshow(title_window, total_image)


def snap_to_time(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if y >= IMG_HEIGHT and y < 2 * IMG_HEIGHT:
            x = x - IMG_WIDTH * 2
            last_tick = (t_end - t_start) * pixels_per_second + first_tick
            if x >= first_tick and x <= last_tick:
                t = ((t_end - t_start) * (x - first_tick) / (last_tick - first_tick)) + t_start
                t_ms = int(t * 1000)
                cv2.setTrackbarPos(trackbar_name, title_window, t_ms)

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

    if flea3_7_timestamps is not None:
        plt.subplot(4,1,1)
        plt.plot(flea3_7_timestamps[:-1], flea3_7_timestamps[1:] - flea3_7_timestamps[:-1])
        plt.xlim([t_start, t_end])
        plt.ylim([0, 3.0/60.0])
        plt.ylabel('{}'.format('flea3_7'))

    if left_camera_timestamps is not None:
        plt.subplot(4,1,2)
        plt.plot(left_camera_timestamps[:-1], left_camera_timestamps[1:] - left_camera_timestamps[:-1])
        plt.xlim([t_start, t_end])
        plt.ylim([0, 3.0/60.0])
        plt.ylabel('{}'.format('left_camera'))

    if right_camera_timestamps is not None:
        plt.subplot(4,1,3)
        plt.plot(right_camera_timestamps[:-1], right_camera_timestamps[1:] - right_camera_timestamps[:-1])
        plt.xlim([t_start, t_end])
        plt.ylim([0, 3.0/60.0])
        plt.ylabel('{}'.format('right_camera'))

    if samsung_mono_timestamps is not None:
        plt.subplot(4,1,4)
        plt.plot(samsung_mono_timestamps[:-1], samsung_mono_timestamps[1:] - samsung_mono_timestamps[:-1])
        plt.xlim([t_start, t_end])
        plt.ylim([0, 3.0/60.0])
        plt.ylabel('{}'.format('samsung_mono'))

    plt.subplot(4,1,1)
    plt.title('depth GT frame time delta (s)')
    ax = plt.subplot(4,1,4)
    plt.xlim([t_start, t_end])
    plt.xlabel('Sequence time (s)')

    plt.tight_layout()

    # https://stackoverflow.com/a/44012582
    xtickslocs = ax.get_xticks()
    ymin, _ = ax.get_ylim()
    xticks_pixels = ax.transData.transform([(xtick, ymin) for xtick in xtickslocs])
    first_tick = xticks_pixels[0][0]
    pixels_per_second = (xticks_pixels[1][0] - xticks_pixels[0][0]) / (xtickslocs[1] - xtickslocs[0])

    canvas.draw()

    width, height = fig.get_size_inches() * fig.get_dpi()
    data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.close()
    plot_bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

    return plot_bgr, first_tick, pixels_per_second


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View all cameras of a sequence with GT depth overlaid to see availability')
    parser.add_argument('--seq', default='scene13_dyn_test_00', help='Sequence name')
    args = parser.parse_args()

    data_base_folder = '/media/levi/EVIMO/npz_extracted'
    file_glob = data_base_folder + '/*/*/*/*'
    files = sorted(list(glob.glob(file_glob)))

    files_grouped_by_sequence = group_files_by_sequence_name(files)
    folders = files_grouped_by_sequence[args.seq][3]

    print('Opening npy files')
    if 'flea3_7' in folders:
        flea3_7_data = np.load(os.path.join(folders['flea3_7'], 'classical.npy'), mmap_mode='r')
        flea3_7_depth = np.load(os.path.join(folders['flea3_7'], 'depth.npy'), mmap_mode='r')
        meta = np.load(os.path.join(folders['flea3_7'], 'meta.npy'), allow_pickle=True).item()
        frame_infos = meta['frames']
        flea3_7_timestamps = np.array([frame_info['cam']['ts'] for frame_info in frame_infos])
    else:
        flea3_7_data = None
        flea3_7_depth = None
        flea3_7_timestamps = None

    if 'left_camera' in folders:
        left_camera_events = np.load(os.path.join(folders['left_camera'], 'events.npy'), mmap_mode='r')
        left_camera_depth = np.load(os.path.join(folders['left_camera'], 'depth.npy'), mmap_mode='r')
        meta = np.load(os.path.join(folders['left_camera'], 'meta.npy'), allow_pickle=True).item()
        frame_infos = meta['frames']
        left_camera_timestamps = np.array([frame_info['cam']['ts'] for frame_info in frame_infos])
    else:
        left_camera_events = None
        left_camera_depth = None
        left_camera_timestamps = None

    if 'right_camera' in folders:
        right_camera_events = np.load(os.path.join(folders['right_camera'], 'events.npy'), mmap_mode='r')
        right_camera_depth = np.load(os.path.join(folders['right_camera'], 'depth.npy'), mmap_mode='r')
        meta = np.load(os.path.join(folders['right_camera'], 'meta.npy'), allow_pickle=True).item()
        frame_infos = meta['frames']
        right_camera_timestamps = np.array([frame_info['cam']['ts'] for frame_info in frame_infos])
    else:
        right_camera_events = None
        right_camera_depth = None
        right_camera_timestamps = None

    if 'samsung_mono' in folders:
        samsung_mono_events = np.load(os.path.join(folders['samsung_mono'], 'events.npy'), mmap_mode='r')
        samsung_mono_depth = np.load(os.path.join(folders['samsung_mono'], 'depth.npy'), mmap_mode='r')
        meta = np.load(os.path.join(folders['samsung_mono'], 'meta.npy'), allow_pickle=True).item()
        frame_infos = meta['frames']
        samsung_mono_timestamps = np.array([frame_info['cam']['ts'] for frame_info in frame_infos])
    else:
        samsung_mono_events = None
        samsung_mono_depth = None
        samsung_mono_timestamps = None

    flea3_7_resolution = (1552, 2080)
    left_camera_resolution  = (480, 640)
    right_camera_resolution = (480, 640)
    samsung_mono_resolution = (480, 640)

    # Can't just use a max because of all the None's
    t_start = None
    t_end = None
    for timestamps in (flea3_7_timestamps, left_camera_timestamps, right_camera_timestamps, samsung_mono_timestamps,
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
    plot_bgr_orig, first_tick_orig, pixels_per_second_orig = make_timestamp_plot(flea3_7_timestamps, left_camera_timestamps, right_camera_timestamps, samsung_mono_timestamps)
    plot_bgr = cv2.resize(plot_bgr_orig, dsize=(IMG_WIDTH, IMG_HEIGHT))
    first_tick = first_tick_orig * (IMG_HEIGHT / plot_bgr_orig.shape[0])
    pixels_per_second = pixels_per_second_orig  * (IMG_HEIGHT / plot_bgr_orig.shape[0])

    title_window = args.seq
    cv2.namedWindow(title_window)

    trackbar_name = 't (ms)'
    slider_max = int(1000 * t_end)
    cv2.createTrackbar(trackbar_name, title_window , int(slider_max/2), slider_max, on_trackbar)
    cv2.setMouseCallback(title_window, snap_to_time)

    print('Visualizing first slider position')
    on_trackbar(int(slider_max/2))

    cv2.waitKey()
