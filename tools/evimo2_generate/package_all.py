#!/usr/bin/python3
import sys, os, shutil, subprocess, argparse

# its a dataset folder is at least one of camera_folders is in it, as well as objectstxt:
camera_folders = set(['flea3_7', 'left_camera', 'right_camera', 'samsung_mono'])
objectstxt = 'objects.txt'
# This will ignore all ground_truth_XXXXXX folders
blacklisted = set(['ground_truth'])
copy_not_move_list = set(['objects.txt', 'calib.txt', 'extrinsics.txt', 'params.txt', '.bag'])

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
        file_missing = False
        for f in self.required:
            if (not os.path.exists(os.path.join(self.folder, f))):
                print ("Required file is missing! skipping sequence/camera!:", self.folder, f)
                file_missing = True
        if file_missing:
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

    def copy(self, dst, dry_run=False): 
        if (not self.check_required()): return False
        src_list  = [os.path.join(self.folder, f) for f in (self.required + self.whitelist)]

        tgt_list = []
        for f in (self.required_tgt + self.whitelist_tgt):
            os.makedirs(os.path.join(dst, os.path.dirname(os.path.normpath(f))), exist_ok=True)
            tgt_list.append(os.path.join(dst, f))

        for i in range(len(src_list)):
            #print (src_list[i], '->', tgt_list[i])
            # Determine if a file should be copied instead of moved
            copy_not_move = False
            for file in copy_not_move_list:
                if file in src_list[i]:
                    copy_not_move = True
                    break

            if copy_not_move:
                if not dry_run:
                    shutil.copyfile(src_list[i], tgt_list[i])
            else:
                if not dry_run:
                    shutil.move(src_list[i], tgt_list[i])

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
    def __init__(self, folder, camera_name, ground_truth):
        super().__init__(folder)

        self.required  = ['objects.txt'] + [os.path.join(camera_name, f_) for f_ in ['calib.txt', 'extrinsics.txt', 'params.txt']]
        self.required_tgt = self.required.copy()

        self.required += [os.path.join(camera_name, ground_truth, f_) for f_ in ['events.txt', 'meta.txt', 'position_plots.pdf']]
        self.required_tgt += ['events.txt', 'meta.txt', 'position_plots.pdf']

        if (not os.path.exists(os.path.join(folder, camera_name, ground_truth))):
            return

        for f in os.listdir(os.path.join(folder, camera_name, ground_truth)):
            if ('.png' in f):
                self.required.append(os.path.join(camera_name, ground_truth, f))
                self.required_tgt.append(os.path.join('img', f))


class NpzSelector(Selector):
    def __init__(self, folder, camera_name, ground_truth):
        super().__init__(folder)

        self.required = [os.path.join(camera_name, ground_truth, 'dataset_events_t.npy'),
                         os.path.join(camera_name, ground_truth, 'dataset_events_xy.npy'),
                         os.path.join(camera_name, ground_truth, 'dataset_events_p.npy'),
                         os.path.join(camera_name, ground_truth, 'dataset_info.npz'),
                         os.path.join(camera_name, ground_truth, 'dataset_depth.npz'),
                         os.path.join(camera_name, ground_truth, 'dataset_mask.npz'),
                         os.path.join(camera_name, ground_truth, 'dataset_classical.npz')]

        self.required_tgt = ['dataset_events_t.npy',
                             'dataset_events_xy.npy',
                             'dataset_events_p.npy',
                             'dataset_info.npz',
                             'dataset_depth.npz',
                             'dataset_mask.npz',
                             'dataset_classical.npz']


class VideoSelector(Selector):
    def __init__(self, folder, camera_name, ground_truth):
        super().__init__(folder)
        self.required = [self.sequence_name + '_' + camera_name + '_' + ground_truth +'.mp4']
        self.required_tgt = self.required.copy()


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
        for b in blacklisted:
            if (b in f): continue
            ret += get_dataset_folders(os.path.join(folder, f))
    return ret

def group_id_from_name(fname, groups):
    spl = fname.split('/')
    for i, g in enumerate(groups):
        if g in spl: return i
    return None

def subgroup_from_name(f):
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
    return subgroup

def move_all(groups, idir, odir, dry_run):
    dataset_folders = sorted(get_dataset_folders(idir))

    for f in dataset_folders:
        gid = group_id_from_name(f, groups)
        if gid is None:
            print (f, "has no group!")
            continue

        g = groups[gid]

        subgroup = subgroup_from_name(f)
        if subgroup is None:
            continue

        print('Moving: ', f)

        # Attempt to copy the raw files into the output directory
        raw = RawSelector(f)
        raw_copy_success = raw.copy(os.path.join(odir, 'raw', g, subgroup, raw.sequence_name), dry_run=dry_run)
        if not raw_copy_success:
            print('Failed to copy raw', f, g)
            continue

        for c in camera_folders:
            if os.path.exists(os.path.join(f, c)):
                potential_ground_truths = os.listdir(os.path.join(f, c))
                # npz and txt run for each ground_truth
                for potential_ground_truth in potential_ground_truths:
                    if 'ground_truth_' in potential_ground_truth:
                        ground_truth = potential_ground_truth
                        npz = NpzSelector(f, c, ground_truth)
                        npz.copy(os.path.join(odir, 'npz', c, g, subgroup, raw.sequence_name+'_'+ground_truth[-6:]), dry_run=dry_run)

                        txt = TxtSelector(f, c, ground_truth)
                        txt.copy(os.path.join(odir, 'txt', c, g, subgroup, raw.sequence_name+'_'+ground_truth[-6:]), dry_run=dry_run)

                        mp4 = VideoSelector(f, c, ground_truth)
                        mp4.copy(os.path.join(odir, 'video', c, g, subgroup), dry_run=dry_run)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('idir',
                        type=str)
    parser.add_argument('odir',
                        type=str)
    parser.add_argument('dry_move',
                        type=str)
    args = parser.parse_args()

    if args.dry_move == 'dry':
        dry_run = True
    elif args.dry_move == 'move':
        dry_run = False
    else:
        raise ValueError('Unknown value set for option dry_move')

    groups = ['imo', 'sanity', 'sfm', 'imo_gaps', 'sanity_gaps', 'sfm_gaps', 'imo_ll', 'sanity_ll', 'sfm_ll']
    move_all(groups, args.idir, args.odir, dry_run)
