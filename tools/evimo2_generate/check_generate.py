#!/usr/bin/python3
import sys, os, subprocess, argparse

# its a dataset folder is at least one of camera_folders is in it, as well as objectstxt:
camera_folders = set(['flea3_7', 'left_camera', 'right_camera', 'samsung_mono'])
objectstxt = 'objects.txt'
blacklisted = set(['ground_truth'])

# Depth and img are to match with depth*.png and img*.png files respectively
txt_files = set(['meta.txt',
                 'events.txt',
                 'position_plots.pdf',
                 'depth',
                 'img'])

npz_files = set(['dataset_classical.npz',
                 'dataset_depth.npz',
                 'dataset_events_p.npy',
                 'dataset_events_xy.npy',
                 'dataset_events_t.npy',
                 'dataset_info.npz',
                 'dataset_mask.npz'])

def get_dataset_folders(folder):
    ret = []
    if (not os.path.isdir(folder)): return ret
    subfolder_names = os.listdir(folder)

    is_df = False
    for cf in camera_folders:
        if cf in subfolder_names: is_df = True
    if objectstxt not in subfolder_names: is_df = False

    if (is_df): ret.append(folder)

    for f in subfolder_names:
        if (f in blacklisted): continue
        ret += get_dataset_folders(os.path.join(folder, f))
    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir',
                        type=str)
    args = parser.parse_args()

    dataset_folders = sorted(get_dataset_folders(args.dir))

    df_no_bagfiles = []
    df_c_no_video = []
    df_c_txt_missing = []
    df_c_npz_missing = []

    seq_lens = {}
    # For every dataset folder
    for df in sorted(dataset_folders):
        if ('_template' in df): continue

        subfolders = os.listdir(df)

        bag_file_name = None
        cameras = {}

        # Get the bag file name and the available cameras
        for sf in subfolders:
            if '.bag' in sf:
                bag_file_name = sf
            if sf in camera_folders:
                cameras[sf] = {}

        if (bag_file_name is None):
            df_no_bagfiles.append(df)

        # Look for the files associated with each camera
        for c in cameras.keys():
            ground_truth_dir = os.path.join(df, c, 'ground_truth')
            
            # If the ground truth dir is missing, 
            if os.path.exists(ground_truth_dir):
                ground_truth_files = os.listdir(ground_truth_dir)

                # Look for txt format files
                for txt_file in txt_files:
                    for ground_truth_file in ground_truth_files:
                        if txt_file in ground_truth_file:
                            cameras[c][txt_file] = txt_file
                    if txt_file not in cameras[c].keys():
                        # Only flea3_7 should have img files
                        if c == 'flea3_7' or not txt_file == 'img':
                            df_c_txt_missing.append((df, c, txt_file))

                # Look for npz files
                for npz_file in npz_files:
                    for ground_truth_file in ground_truth_files:
                        if npz_file in ground_truth_file:
                            cameras[c][npz_file] = npz_file
                    if npz_file not in cameras[c].keys():
                        df_c_npz_missing.append((df, c, npz_file))


                # Look for video
                for sf in subfolders:
                    if c + '.mp4' in sf:
                        cameras[c]['mp4'] = sf

                if 'mp4' not in cameras[c].keys():
                    df_c_no_video.append((df, c))

            # Ground truth dir is missing, so all files are missing
            else:
                for txt_file in txt_files:
                    if c == 'flea3_7' or not txt_file == 'img':
                        df_c_txt_missing.append((df, c, txt_file))

                for npz_file in npz_files:
                    df_c_no_video.append((df, c))


        # Print what files were found for each camera in the dataset folder
        print (df)
        print ("\tbag file:\t", bag_file_name)
        for c in sorted(cameras.keys()):
            s = "\t" + str(c) + ":\t"

            for key in sorted(cameras[c].keys()):
                s += '\n\t\t' + str(cameras[c][key])

            print (s)

    if (len(df_no_bagfiles) > 0):
        print ("\n {} folders without .bag files:".format(len(df_no_bagfiles)))
        for df in sorted(df_no_bagfiles):
            print("\t", df)
    else:
        print ("\n No missing .bag files")

    if (len(df_c_txt_missing) > 0):
        print ("\n {} missing txt format files:".format(len(df_c_txt_missing)))
        for df in sorted(df_c_txt_missing):
            print("\t", df)
    else:
        print ("\n No missing txt format files")

    if (len(df_c_npz_missing) > 0):
        print ("\n {} missing npz format files:".format(len(df_c_npz_missing)))
        for df in sorted(df_c_npz_missing):
            print("\t", df)
    else:
        print ("\n No missing npz format files")

    if (len(df_c_no_video) > 0):
        print ("\n {} missing videos:".format(len(df_c_no_video)))
        for df in sorted(df_c_no_video):
            print("\t", df)
    else:
        print ("\n No missing videos")
