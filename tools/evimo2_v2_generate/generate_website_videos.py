#!/usr/bin/python3
import argparse
import os
import glob
import subprocess
import tqdm

def get_sequence_name(full_file_name):
    file_name = os.path.split(full_file_name)[1]
    video_name = os.path.splitext(file_name)[0]

    if 'flea3_7_ground_truth_' in video_name:
        left, right = video_name.split('flea3_7_ground_truth_')
    elif 'left_camera_ground_truth_' in video_name:
        left, right = video_name.split('left_camera_ground_truth_')
    elif 'right_camera_ground_truth_' in video_name:
        left, right = video_name.split('right_camera_ground_truth_')
    elif 'samsung_mono_ground_truth_' in video_name:
        left, right = video_name.split('samsung_mono_ground_truth_')
    else:
        raise Exception('should not happen')

    sequence_name = left + right

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stitch generated videos to together for website previews')
    parser = argparse.ArgumentParser()
    parser.add_argument('--idir', type=str, help='Directory containing video file tree')
    parser.add_argument('--odir', type=str, help='Directory where videos will be put')
    args = parser.parse_args()

    data_base_folder = args.idir
    file_glob = data_base_folder + '/*/*/*/*'
    files = sorted(list(glob.glob(file_glob)))

    files_grouped_by_sequence = group_files_by_sequence_name(files)
    #folders = files_grouped_by_sequence[args.seq][3]

    for seq in tqdm.tqdm(files_grouped_by_sequence,  dynamic_ncols=True):
    #for seq in ['scene16_d_dyn_test_01_000001',]:
        videos = files_grouped_by_sequence[seq][3]

        inputs = 0
        to_tile = []

        def add_video(name):
            global inputs
            if name in videos:
                to_tile.append([str(inputs), videos[name], name])
                inputs+=1
            else:
                to_tile.append(('bg', None, name))

        add_video('flea3_7')
        add_video('left_camera')
        add_video('right_camera')
        add_video('samsung_mono')

        def make_ffmpeg_cmd(videos, output_name):
            ffmpeg_cmd = ['ffmpeg', '-hide_banner', '-loglevel' , 'error']

            num_inputs = 0
            for video in videos:
                if video[1] is not None:
                    ffmpeg_cmd += ('-i', '{}'.format(video[1]))

            left_camera_id = 'left'
            right_camera_id = 'right'
            samsung_mono_id = 'sam'
            filter_complex = ''
            for video in videos:
                if video[2] == 'flea3_7':
                    if video[1] is not None:
                        filter_complex += '[{}]scale=1280:480[flea3];'.format(video[0])
                    else:
                        filter_complex += 'color=s=1280x480:c=black[flea3];'
                elif video[2] == 'left_camera':
                    if video[1] is None:
                        filter_complex += 'color=s=1280x480:c=black[left];'
                    else:
                        left_camera_id=video[0]
                elif video[2] == 'right_camera':
                    if video[1] is None:
                        filter_complex += 'color=s=1280x480:c=black[right];'
                    else:
                        right_camera_id=video[0]
                elif video[2] == 'samsung_mono':
                    if video[1] is None:
                        filter_complex += 'color=s=1280x480:c=black[sam];'
                    else:
                        samsung_mono_id=video[0]
                else:
                    raise Exception('should not happen')

            filter_complex += ('[flea3][{}]hstack=shortest=1[t];[{}][{}]hstack=shortest=1[b];[t][b]vstack=shortest=1[v]'
                .format(samsung_mono_id, left_camera_id, right_camera_id))

            ffmpeg_cmd += ('-filter_complex', filter_complex)
            ffmpeg_cmd += ('-map', '[v]')
            ffmpeg_cmd += (output_name,)
            return ffmpeg_cmd

        category = files_grouped_by_sequence[seq][0]

        output_name = os.path.join(args.odir, category + '_' + seq + '.mp4')
        ffmpeg_cmd = make_ffmpeg_cmd(to_tile, output_name)

        #print(seq)
        encode_process = subprocess.Popen(ffmpeg_cmd)
        encode_process.wait()
