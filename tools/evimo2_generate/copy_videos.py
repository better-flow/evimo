#!/usr/bin/python3
import sys, os, shutil, subprocess, argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('idir',
                        type=str)
    args = parser.parse_args()
    idir = args.idir

    videos_dir = os.path.join(idir, 'VIDEOS')
    groups = ['imo', 'sanity', 'sfm', 'imo_gaps', 'sanity_gaps', 'sfm_gaps', 'imo_ll', 'sanity_ll', 'sfm_ll']

    for g in os.listdir(os.path.join(idir)):
        if (g not in groups): continue

        for subgroup in os.listdir(os.path.join(idir, g)):
            for dtype in os.listdir(os.path.join(idir, g, subgroup)):
                if (dtype == 'video'):
                    src = os.path.join(idir, g, subgroup, dtype)
                    dst = os.path.join(videos_dir, g, subgroup, dtype)
                    os.makedirs(dst, exist_ok=True)
                    shutil.copytree(src, dst, dirs_exist_ok=True)
