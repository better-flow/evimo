import argparse
import os
import cv2
import numpy as np
import pprint
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from matplotlib import pyplot as plt

def sequence_flow_histogram(folder, max_flow=400):
    flow_npz = np.load(os.path.join(folder, 'dataset_flow.npz'))
    flow_names = sorted([f for f in flow_npz if 'flow_' in f])

    hist = None
    bin_edges = None
    max_comp = 0
    for flow_name in tqdm(flow_names, position=1, desc='{}'.format(os.path.split(folder)[1])):
        flow = flow_npz[flow_name]
        flow[np.isnan(flow)] = 0

        new_max = np.max(flow)
        if new_max > max_comp:
            max_comp = new_max

        flow[flow > max_flow] = max_flow

        new_hist, new_bin_edges = np.histogram(flow, bins=max_flow, range=[0, max_flow])
        assert new_bin_edges is not None

        if bin_edges is None:
            bin_edges = new_bin_edges
            hist = new_hist
        else:
            assert np.all(bin_edges == new_bin_edges)
            hist += new_hist

    assert bin_edges is not None

    return (os.path.split(folder)[1], max_comp), bin_edges, hist, folder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*',help='Sequences to analyze')

    args = parser.parse_args()
    sequence_folders=args.files

    with Pool(processes=multiprocessing.cpu_count()) as p:
        results = p.map(sequence_flow_histogram, sequence_folders)

    max_vals = []
    bin_edges = None
    hist = None
    for max_val_pair, new_bin_edges, new_hist, folder in results:
        max_vals.append(max_val_pair)

        if bin_edges is None:
            bin_edges = new_bin_edges
            hist = new_hist
        else:
            assert np.all(bin_edges == new_bin_edges)
            hist += new_hist

    pprint.pprint(max_vals)

    plt.bar(bin_edges[:-1], hist)
    plt.yscale('log')
    plt.xlabel('x or y displacement (pixels)')
    plt.ylabel('Number of samples')
    plt.grid()
    plt.show()
