#!/usr/bin/python3
import argparse
import os
import glob
import subprocess
import tqdm
from jinja2 import Environment, FileSystemLoader

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

def generate_table(purpose, seqs_urls):
    env = Environment()
    env = Environment(loader=FileSystemLoader('.'))
    tmpl = env.get_template('video_website_table_template.html')

    output = tmpl.render(purpose=purpose,
                         seqs_urls=seqs_urls)
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stitch generated videos to together for website previews')
    parser = argparse.ArgumentParser()
    parser.add_argument('--idir', type=str, help='Directory containing video file tree')
    parser.add_argument('--odir', type=str, help='Directory where html will be put')
    args = parser.parse_args()

    data_base_folder = args.idir
    file_glob = data_base_folder + '/*/*/*/*'
    files = sorted(list(glob.glob(file_glob)))

    files_grouped_by_sequence = group_files_by_sequence_name(files)
    #folders = files_grouped_by_sequence[args.seq][3]

    table_data = {}

    for seq in tqdm.tqdm(files_grouped_by_sequence,  dynamic_ncols=True):
        category = files_grouped_by_sequence[seq][0]
        purpose = files_grouped_by_sequence[seq][1]
        videos = files_grouped_by_sequence[seq][3]
        video_file_name = category + '_' + seq + '.mp4'
        video_url = 'https://obj.umiacs.umd.edu/evimo2v2videowebsite/' + video_file_name

        if category == 'sanity' or category == 'sanity_ll':
            purpose = 'eval'

        if not category in table_data:
            table_data[category] = {}

        if not purpose in table_data[category]:
            table_data[category][purpose] = []

        table_data[category][purpose].append((seq, video_url))

    for category in table_data:
        for purpose in table_data[category]:
            output = generate_table(purpose, table_data[category][purpose])
            file_name = category + '_' + purpose + '.html'
            print(file_name)

            file = os.path.join(args.odir, file_name)

            with open(file, 'w') as f:
                f.write(output)
