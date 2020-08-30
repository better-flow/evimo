#!/usr/bin/python3

import os, sys, time, shutil
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt



def get_blobs(img_):
    if (len(img_.shape) >= 3):
        img = img_[:,:,0].copy()
    else:
        img = img_.copy()
    img = img.astype(np.uint8)
    img = 255 - img

    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 80
    params.maxThreshold = 100

    # Filter by Area.
    params.filterByArea = True
    params.minArea = img.shape[0] * img.shape[1] * 2e-6
    params.maxArea = img.shape[0] * img.shape[1] * 2e-4

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.7

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(img)
    ret = np.zeros(shape=(len(keypoints), 3), dtype=np.float32)

    for i, k in enumerate(keypoints):
        ret[i,:2] = k.pt
        ret[i,2]  = k.size

    #im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #plt.imshow(im_with_keypoints)


    #plt.show()
    return ret


def find_all_3lines(keypoints, th):
    print ("Searchong for 3-lines...")
    n_pts = keypoints.shape[0]
    ret = []
    err = []
    for i in range(n_pts - 2):
        for j in range(i + 1, n_pts - 1):
            dp = keypoints[i,:2] - keypoints[j,:2]
            l_dp = np.linalg.norm(dp)
            cross = keypoints[i,0] * keypoints[j,1] - keypoints[i,1] * keypoints[j,0]
            if (l_dp < th): continue

            for k in range(j + 1, n_pts):
                d = np.abs(dp[1] * keypoints[k,0] - dp[0] * keypoints[k,1] + cross) / l_dp
                if (d >= th): continue

                points = np.vstack((keypoints[i,:2], keypoints[j,:2], keypoints[k,:2]))

                if (dp[0] > dp[1]):
                    order = np.argsort(points[:,0])
                else:
                    order = np.argsort(points[:,1])

                idx = np.array([i,j,k])
                ret.append(idx[order])
                err.append(d)

                print('\t', idx[order],'\t', d)
    return np.array(ret), np.array(err)


def plot_lines(img_, keypoints, idx):
    if (len(img_.shape) >= 3):
        img = np.dstack((img_, img_, img_)).copy()
    else:
        img =img_.copy()

    for i, line in enumerate(idx):
        p0 = keypoints[line[0],:2].astype(np.int32)
        p1 = keypoints[line[1],:2].astype(np.int32)
        p2 = keypoints[line[2],:2].astype(np.int32)

        img = cv2.line(img, tuple(p0), tuple(p1), (255,0,0), 2)
        img = cv2.line(img, tuple(p1), tuple(p2), (0,255,0), 2)

    plt.imshow(img)


def detect_wand(keypoints, idx, wand_3d_mapping, th_rel, th_lin, th_ang, img_=None):
    print ("Detecting wand...")
    if (img_ is None):
        img = None
    elif (len(img_.shape) >= 3):
        img = img_.copy()
    else:
        img = np.dstack((img_, img_, img_)).copy()

    ret = []
    for i, l1 in enumerate(idx):
        for j, l2 in enumerate(idx):
            if (i == j): continue
            if (l1[1] != l2[0] and l1[1] != l2[2]): continue
            if (len(set(l1).union(l2)) != 5): continue

            p2 = keypoints[l1[1],:2] # center point

            d1 = np.linalg.norm(p2 - keypoints[l1[0],:2])
            d2 = np.linalg.norm(p2 - keypoints[l1[2],:2])
            if (d1 > d2):
                p1 = keypoints[l1[0],:2]
                p3 = keypoints[l1[2],:2]
            else:
                p1 = keypoints[l1[2],:2]
                p3 = keypoints[l1[0],:2]

            p4 = keypoints[l2[1],:2]

            if (l1[1] == l2[0]):
                p5 = keypoints[l2[2],:2]
            else:
                p5 = keypoints[l2[0],:2]

            gt_rel1 = np.linalg.norm(wand_3d_mapping['red'][1] - wand_3d_mapping['red'][0]) / \
                                     np.linalg.norm(wand_3d_mapping['red'][1] - wand_3d_mapping['red'][2])
            rel1 = np.linalg.norm(p2 - p1) / np.linalg.norm(p2 - p3)
            if (abs(gt_rel1 - rel1) > th_rel): continue

            rel2 = np.linalg.norm(p4 - p2) / np.linalg.norm(p4 - p5)
            gt_rel2 = np.linalg.norm(wand_3d_mapping['red'][3] - wand_3d_mapping['red'][1]) / \
                                     np.linalg.norm(wand_3d_mapping['red'][3] - wand_3d_mapping['red'][4])
            if (abs(rel2 - gt_rel2) > th_lin): continue

            rel3 = np.linalg.norm(p1 - p3) / np.linalg.norm(p2 - p5)
            gt_rel3 = np.linalg.norm(wand_3d_mapping['red'][0] - wand_3d_mapping['red'][2]) / \
                                     np.linalg.norm(wand_3d_mapping['red'][1] - wand_3d_mapping['red'][4])
            if (abs(rel3 - gt_rel3) > th_lin): continue

            v1 = p2 - p1
            v2 = p2 - p4
            angle = abs(np.arctan2(v1[0], v1[1]) - np.arctan2(v2[0], v2[1]))
            angle = min(abs(angle), abs(angle - 2 * np.pi), abs(angle + 2 * np.pi))
            if (abs(angle - np.pi / 2) > th_ang): continue

            print ("\tFound:", l1, l2)
            ret.append(np.vstack((p1, p2, p3, p4, p5)))

            if (img is None): continue
            img = cv2.line(img, tuple(p2.astype(np.int32)), tuple(p1.astype(np.int32)), (255,0,0), 2)
            img = cv2.line(img, tuple(p2.astype(np.int32)), tuple(p3.astype(np.int32)), (0,255,0), 2)
            img = cv2.line(img, tuple(p2.astype(np.int32)), tuple(p4.astype(np.int32)), (0,0,255), 2)
            img = cv2.line(img, tuple(p4.astype(np.int32)), tuple(p5.astype(np.int32)), (255,255,255), 2)

            plt.imshow(img)
            plt.show()

    if (len(ret) > 1):
        print ("Failed: Found more than 1 candidates")
        return None
    if (len(ret) == 0):
        print ("Failed: Found 0 candidates")
        return None
    return ret[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get some data.')
    parser.add_argument('f', type=str, help='root dir', default="")
    args = parser.parse_args()

    image = cv2.imread(args.f)

    keypoints = get_blobs(image)

    print ("Keypoints:")
    print (keypoints)

    wand_3d_mapping = {'red': np.array([[30.576542, -150.730270, -45.588951],
                                        [-45.614960, -18.425253, 2.359259],
                                        [-83.571022, 47.522499, 26.421692],
                                        [64.297485, 39.157578, 6.395612],
                                        [169.457520, 97.256927, 11.720362]]),
                        'ir': np.array([[32.454151, -153.981064, -46.729141],
                                        [-43.735237, -21.440924, 1.182336],
                                        [-81.761475, 44.537903, 25.208614],
                                        [60.946213, 36.995384, 6.158191],
                                        [166.117447, 95.441071, 11.552736]])}

    idx, err = find_all_3lines(keypoints, th=max(image.shape[0], image.shape[1]) * 5e-3)
    wand_points = detect_wand(keypoints, idx, wand_3d_mapping, th_rel=0.5, th_lin=0.5, th_ang=0.5, img_=image)



