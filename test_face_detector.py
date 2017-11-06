import argparse
import glob
import os
import sys
import progressbar
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import imread
from scipy.misc import imsave
from face_detection import find_faces, find_faces_single_scale, find_faces_single_scale_max_supp

# Princeton University, COS 429, Fall 2017
#
# test_single_scale.m
# Test face detection on all single-scale images
# Inputs:
# stride: how far to move between locations at which the detector is run
# thresh: probability threshold for calling a detection a face

def test_multi_scale(stride, thresh, show_plots=True):
    params = np.load("face_classifier_params.npy")
    multi_scale_scenes_dir = "../cos429_f17_assignment2_part4/face_data/testing_scenes"
    scene_filenames = glob.glob(multi_scale_scenes_dir + '/*.jpg')

    for i in range(1, len(scene_filenames)):
        path = os.path.abspath(scene_filenames[i])
        filename = os.path.basename(path)
        sys.stdout.write("Detecting faces in {}...\n".format(filename))
        sys.stdout.flush()
        img = imread(path, mode='L')
        out_img = find_faces(img, stride, thresh, params, 9, False)
        imsave(filename + '_out.jpg', out_img)
        if show_plots:
            plt.imshow(out_img, cmap='gray')
            plt.show()
            choice = input("Continue? (y/n): ")
            if choice == 'n':
                return

def test_single_scale(stride, thresh, show_plots=True):
    params = np.load("face_classifier_params.npy")
    single_scale_scenes_dir = "../cos429_f17_assignment2_part3/face_data/single_scale_scenes"
    scene_filenames = glob.glob(single_scale_scenes_dir + '/*.jpg')

    for i in range(len(scene_filenames)):
        path = os.path.abspath(scene_filenames[i])
        filename = os.path.basename(path)
        sys.stdout.write("Detecting faces in {}...\n".format(filename))
        sys.stdout.flush()
        img = imread(path, mode='L')
        out_img, prob_map = find_faces_single_scale_max_supp(img,
                                                    stride,
                                                    thresh,
                                                    params,
                                                    9,
                                                    False)
        imsave(filename + '_single_out.jpg', out_img)
        imsave(filename + '_single_prob.jpg', prob_map)
        if show_plots:
            plt.imshow(out_img, cmap='gray')
            plt.show()
            plt.imshow(prob_map, cmap='gray')
            plt.show()
            choice = input("Continue? (y/n): ")
            if choice == 'n':
                return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Face Detection")
    parser.add_argument('-s', help="stride", dest="stride", type=int, default=3)
    parser.add_argument('-t', help="threshhold", dest="thresh", type=float, default=0.95)
    parser.add_argument('-m', help="use multi-scale", dest="multi_scale", type=int, default=0)
    parser.add_argument('-p', help="show plots", dest="plot", type=int, default=1)
    args = parser.parse_args()
    show_plot = True if args.plot else False
    if args.multi_scale:
        test_multi_scale(args.stride, args.thresh, show_plots=show_plot)
    else:
        test_single_scale(args.stride, args.thresh, show_plots=show_plot)
