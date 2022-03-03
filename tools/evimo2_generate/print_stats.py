#!/usr/bin/python3
import sys, os, subprocess, argparse


# its a dataset folder is at least one of camera_folders is in it, as well as objectstxt:
camera_folders = set(['flea3_7', 'left_camera', 'right_camera', 'samsung_mono'])
objectstxt = 'objects.txt'
blacklisted = set(['ground_truth'])


def exec_command(cmd_, silent=False):
    if not silent:
        print ("Executing bash commands:")
        print (cmd_+"...")
    try:
        result = subprocess.check_output(["bash", "-c", cmd_])
        if not silent:
            print (result)
   
    except subprocess.CalledProcessError as e:
        print ("Error while executing '"+str(e.cmd)+
                      "': the return code is "+str(e.returncode)+": "+str(e.output))
        print ("If you want to return to this place restart the script.")
        return [1, ""]

    except:
        print ("Something has went wrong! (" + str(sys.exc_info()) + ")")
        print ("If you want to return to this place restart the script.")
        return [1, ""]
    return [0, result]


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



def genrate_gt(f, cam, fps=40, t_offset=0, t_len=-1):
    print ("Generating gt for", f, cam)
    cmd  = "roslaunch evimo event_imo_offline.launch"
    cmd += " show:=-1"
    cmd += " folder:=" + str(f)
    cmd += " camera_name:="+ str(cam)
    cmd += " generate:=true"
    cmd += " save_3d:=false"
    cmd += " fps:=" + str(fps)
    cmd += " t_offset:=" + str(t_offset)
    cmd += " t_len:=" + str(t_len)
    exec_command(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir',
                        type=str)
    args = parser.parse_args()

    dataset_folders = sorted(get_dataset_folders(args.dir))

    #genrate_gt(dataset_folders[0], sorted(camera_folders)[0])
    #exit(-1)


    for f in dataset_folders:
        print (f)

    print()
    df_no_bagfiles = []
    df_to_regenerate = []
    df_partial_regenerate = {}

    seq_lens = {}
    for df in sorted(dataset_folders):
        if ('_template' in df): continue

        subfolders = os.listdir(df)

        bag_file = None
        cameras = {}
        for sf in subfolders:
            if '.bag' in sf:
                bag_file = sf
            if (sf in camera_folders):
                cameras[sf] = {}

        for c in cameras.keys():
            for sf in subfolders:
                if (c in sf) and ('.mp4' in sf):
                    cameras[c]['mp4'] = sf
            c_folder = os.path.join(df, c)
            if (not os.path.exists(c_folder)): continue

            if (c_folder is None): continue
            meta_path = os.path.join(c_folder, 'ground_truth', 'meta.txt')
            if (not os.path.exists(meta_path)): continue
            continue
            meta = eval(open(meta_path).read())
            cameras[c]['len'] = meta['frames'][-1]['ts'] - meta['frames'][0]['ts']
            if (c not in seq_lens.keys()): seq_lens[c] = 0.0
            seq_lens[c] += cameras[c]['len']

        print ()
        print (df)
        print ("\tbag file:\t", bag_file)
        for c in cameras.keys():
            s = "\t\t" + str(c) + ":\t"
            if ('mp4' in cameras[c].keys()): s+=str(cameras[c]['mp4']) + '\t'
            if ('len' in cameras[c].keys()): s+=str(cameras[c]['len'])
            print (s)

        if (bag_file is None):
            df_no_bagfiles.append(df)

        complete_regen = True
        for c in cameras.keys():
            if ('mp4' in cameras[c].keys()): complete_regen = False
        if (complete_regen): df_to_regenerate.append(df)

        if (not complete_regen):
            for c in cameras.keys():
                if ('mp4' in cameras[c].keys()): continue
                if df not in df_partial_regenerate.keys(): df_partial_regenerate[df] = []
                df_partial_regenerate[df].append(c)


    if (len(df_no_bagfiles) > 0):
        print ("\nFolders without .bag files:")
        for df in sorted(df_no_bagfiles):
            print("\t", df)

    if (len(df_to_regenerate) > 0):
        print ("\nFolders without ground truth at all:")
        for df in sorted(df_to_regenerate):
            print("\t", df)

    if (len(df_partial_regenerate.keys()) > 0):
        print ("\nFolders with ground truth missing for some cameras:")
        for df in sorted(df_partial_regenerate.keys()):
            print("\t", df, ':', sorted(df_partial_regenerate[df]))

    print ("\nPer-camera recording lengths:")
    for c in sorted(seq_lens.keys()):
        print ("\t", c, ':', seq_lens[c])
