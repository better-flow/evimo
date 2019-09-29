#!/usr/bin/python3

import argparse
import numpy as np
import os, sys, shutil, signal, glob, time
import matplotlib.colors as colors
import pydvs, cv2


global_scale_pn = 1
global_scale_pp = 1
global_shape = (480, 640)
slice_width = 1


def clear_dir(f):
    if os.path.exists(f):
        print ("Removed directory: " + f)
        shutil.rmtree(f)
    os.makedirs(f)
    print ("Created directory: " + f)


def dvs_img(cloud, shape, K, D):
    cmb = pydvs.dvs_img(cloud, shape, K=K, D=D)

    cmb[:,:,0] *= global_scale_pp
    cmb[:,:,1] *= 255.0 / slice_width
    cmb[:,:,2] *= global_scale_pn

    return cmb
    #return cmb.astype(np.uint8)


def nz_avg(img_):
    img = np.copy(img_)
    img[img < 1.0] = np.nan
    return np.nanmean(img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        type=str,
                        default='.',
                        required=True)
    parser.add_argument('--nbins',
                        type=int,
                        required=False,
                        default=1)

    args = parser.parse_args()

    print ("Opening", args.base_dir)

    sl_npz = np.load(args.base_dir + '/recording.npz')
    cloud          = sl_npz['events']
    idx            = sl_npz['index']
    discretization = sl_npz['discretization']
    K = None
    D = None

    slice_width = args.nbins * discretization

    first_ts = cloud[0][0]
    last_ts = cloud[-1][0]

    vis_dir   = os.path.join(args.base_dir, 'vis')
    pydvs.replace_dir(vis_dir)

    print ("The recording range:", first_ts, "-", last_ts)
    print ("Discretization resolution:", discretization)

    import matplotlib.pyplot as plt
    ebins = []
    for i in range(len(idx) - args.nbins):
        nevents = idx[i + args.nbins] - idx[i]
        ebins.append(nevents)

    median = np.median(ebins)
    cutoff = [median for b in ebins]
    for i, b in enumerate(ebins):
        lo = max(0, i - args.nbins * 10)
        hi = min(len(ebins), i + args.nbins * 10)
        cutoff[i] = np.mean(ebins[lo:hi])

    mpeaks = [0 for b in ebins]
    current_peak = []
    for i, b in enumerate(ebins):
        if (cutoff[i] < b):
            current_peak.append(b)
            continue

        if (len(current_peak) == 0):
            continue

        m_peak = current_peak.index(max(current_peak))
        mpeaks[i - len(current_peak) + m_peak] = max(current_peak)
        current_peak = []


    fig = plt.figure()
    plt.rc('lines', linewidth=1.0)

    plt.plot(ebins)
    plt.plot(cutoff)
    plt.plot(mpeaks)

    fig.set_size_inches(50, 20)
    plt.savefig(os.path.join(args.base_dir, 'event_rate_plot.png'), dpi=300, bbox_inches='tight')
    #plt.show()

    id_ = 0
    for i, b in enumerate(mpeaks):
        if (mpeaks[i] <= 1):
            continue;

        time = i * discretization
        if (time > last_ts or time < first_ts):
            continue

        sl, _ = pydvs.get_slice(cloud, idx, time, slice_width, 1, discretization)

        eimg = dvs_img(sl, global_shape, K, D)
        cimg = eimg[:,:,0] + eimg[:,:,2]

        avg = nz_avg(cimg)
        cimg *= 127 / avg
        #cimg *= 50
        cimg = cimg.astype(np.uint8)

        cv2.imwrite(os.path.join(vis_dir, 'frame_' + str(id_).rjust(10, '0') + '.png'), cimg)
        id_ += 1
