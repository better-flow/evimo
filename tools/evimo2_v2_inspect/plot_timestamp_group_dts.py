import matplotlib.pyplot as plt
import numpy as np

index2stat = ['min', 'max', 'avg', 'std', 'median']
cameras = ['left_camera', 'right_camera', 'samsung_mono']

if __name__ == '__main__':
    for j in range(len(index2stat)):
        plt.figure()
        for i, camera_name in enumerate(cameras):
            stats = np.load(camera_name + '_stats.npy')
            stat = stats[:, j]
            stat_sorted = np.sort(stat)

            plt.subplot(1, len(cameras), i+1)
            plt.plot(stat_sorted)
            plt.grid()
            plt.title(camera_name)

            if i == 0:
                plt.ylabel('{} time between event groups (seconds)'.format(index2stat[j]))

            if i == 1:
                plt.xlabel('Sequences sorted by statistic ({})'.format(index2stat[j]))

        plt.tight_layout()
    plt.show()
