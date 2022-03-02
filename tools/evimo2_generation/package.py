#!/usr/bin/python3
import sys, os, shutil, subprocess, argparse
from multiprocessing import Pool
import tarfile

from print_stats import *


def group_id_from_name(fname, groups):
    spl = fname.split('/')
    for i, g in enumerate(groups):
        if g in spl: return i
    return None



class Selector:
    def __init__(self, folder):
        self.folder = folder
        self.files = os.listdir(self.folder)
        self.sequence_name = os.path.basename(os.path.normpath(self.folder))
        self.whitelist = []
        self.required  = []
        self.required_tgt  = []
        self.whitelist_tgt = []

    def check_required(self):
        for f in self.required:
            print (f)
            if (not os.path.exists(os.path.join(self.folder, f))):
                print ("Required file is missing!:", self.folder, f)
                return False
        wlst = []
        wlst_tgt = []
        for i, f in enumerate(self.whitelist):
            if (os.path.exists(os.path.join(self.folder, f))):
                wlst.append(f)
                wlst_tgt.append(self.whitelist_tgt[i])
        self.whitelist = wlst
        self.whitelist_tgt = wlst_tgt
        return True

    def copy(self, dst): 
        if (not self.check_required()): return False
        src_list  = [os.path.join(self.folder, f) for f in (self.required + self.whitelist)]

        tgt_list = []
        for f in (self.required_tgt + self.whitelist_tgt):
            os.makedirs(os.path.join(dst, os.path.dirname(os.path.normpath(f))), exist_ok=True)
            tgt_list.append(os.path.join(dst, f))

        for i in range(len(src_list)):
            print (src_list[i], '->', tgt_list[i])
            shutil.copyfile(src_list[i], tgt_list[i])

        return True


class RawSelector(Selector):
    def __init__(self, folder):
        super().__init__(folder)
        self.required = ['objects.txt', self.sequence_name + '.bag']
        self.whitelist = ['run.sh']
        for f in self.files:
            if (f not in camera_folders): continue
            self.required += [os.path.join(f, f_) for f_ in ['calib.txt', 'extrinsics.txt', 'params.txt']]
        self.required_tgt  = self.required
        self.whitelist_tgt = self.whitelist


class TxtSelector(Selector):
    def __init__(self, folder, camera_name):
        super().__init__(folder)

        self.required  = ['objects.txt'] + [os.path.join(camera_name, f_) for f_ in ['calib.txt', 'extrinsics.txt', 'params.txt']]
        self.required_tgt = self.required.copy()

        self.required += [os.path.join(camera_name, 'ground_truth', f_) for f_ in ['events.txt', 'meta.txt', 'position_plots.pdf']]
        self.required_tgt += ['events.txt', 'meta.txt', 'position_plots.pdf']

        if (not os.path.exists(os.path.join(folder, camera_name, 'ground_truth'))):
            return

        for f in os.listdir(os.path.join(folder, camera_name, 'ground_truth')):
            if ('.png' in f):
                self.required.append(os.path.join(camera_name, 'ground_truth', f))
                self.required_tgt.append(os.path.join('img', f))


class NpzSelector(Selector):
    def __init__(self, folder, camera_name):
        super().__init__(folder)
        self.required = [os.path.join(camera_name, 'ground_truth', 'dataset.npz')]
        self.required_tgt = [self.sequence_name + '.npz']


class VideoSelector(Selector):
    def __init__(self, folder, camera_name):
        super().__init__(folder)
        self.required = [self.sequence_name + '_' + camera_name + '.mp4']
        self.required_tgt = self.required.copy()


def compress(fin, fout, fname):
    print ("compressing", fin, '->', fout, fname)
    os.makedirs(fout, exist_ok=True)

    with tarfile.open(os.path.join(fout, fname), "w:gz") as tar:
        tar.add(fin, arcname=os.path.basename(fin))

    print ("\tdone", fin, '->', fout, fname)



def move_all(groups, idir, odir):
    dataset_folders = sorted(get_dataset_folders(idir))

    for f in dataset_folders:
        gid = group_id_from_name(f, groups)
        if (gid is None):
            print (f, "has no group!")
            continue
        g = groups[gid]
        subgroup = None
        if ('train' in f):
            subgroup = 'train'
        elif ('eval' in f):
            subgroup = 'eval'
        elif ('checkerboard' in f):
            subgroup = 'checkerboard'        
        elif ('depth_var' in f):
            subgroup = 'depth_var'
        elif ('tabletop' in f):
            subgroup = 'tabletop'
        elif ('sliding' in f):
            subgroup = 'sliding'
        else: continue

        print (f, g)

        raw = RawSelector(f)
        if (not raw.copy(os.path.join(odir, 'raw', g, subgroup, raw.sequence_name))): continue
        shutil.copyfile(os.path.join(idir, 'generate.sh'), os.path.join(odir, 'raw', g, subgroup, 'generate.sh'))

        for c in camera_folders:
            npz = NpzSelector(f, c)
            npz.copy(os.path.join(odir, 'npz', c, g, subgroup))

            mp4 = VideoSelector(f, c)
            mp4.copy(os.path.join(odir, 'video', c, g, subgroup))

            txt = TxtSelector(f, c)
            txt.copy(os.path.join(odir, 'txt', c, g, subgroup, raw.sequence_name))



def compress_all_list(idir):

    compressed_dir = os.path.join(idir, 'COMPRESSED')
    os.makedirs(compressed_dir, exist_ok=True)

    fin_list = []
    fout_list = []
    fname_list = []

    for g in os.listdir(os.path.join(idir)):
        for subgroup in os.listdir(os.path.join(idir, g)):
            for dtype in os.listdir(os.path.join(idir, g, subgroup)):
                if (dtype == 'raw'):
                    fin_list.append(os.path.join(idir, g, subgroup, dtype))
                    fout_list.append(os.path.join(compressed_dir, g, subgroup))
                    fname_list.append(subgroup + '_' + g + '_' + dtype + '.tar.gz')
                    continue

                for c in os.listdir(os.path.join(idir, g, subgroup, dtype)):
                    fin_list.append(os.path.join(idir, g, subgroup, dtype, c))
                    fout_list.append(os.path.join(compressed_dir, g, subgroup, dtype))
                    fname_list.append(subgroup + '_' + g + '_' + c + '_' + dtype + '.tar.gz')


    return fin_list, fout_list, fname_list




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('idir',
                        type=str)
    parser.add_argument('odir',
                        type=str)
    args = parser.parse_args()

    groups = ['imo', 'sanity', 'sfm', 'imo_gaps', 'sanity_gaps', 'sfm_gaps', 'imo_ll', 'sanity_ll', 'sfm_ll']
    #groups = ['sfm', 'sfm_gaps', 'sfm_ll']

    move_all(groups, args.idir, args.odir)

    if (False):
        fin_list, fout_list, fname_list = compress_all_list(args.odir)

        print ("To be compressed:")
        for i in range(len(fin_list)):
            print ("\t", fin_list[i], '->', fout_list[i], ':\t', fname_list[i])

        def f(i):
            compress(fin_list[i], fout_list[i], fname_list[i])

        

        #with Pool(12) as p:
        #    p.map(f, range(len(fin_list)))

