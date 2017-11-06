import sys
import progressbar
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave, imresize
from logistic_regression import logistic_prob
from hog36 import hog36
# %
# % Princeton University, COS 429, Fall 2017
# %
# % find_faces_single_scale.m
# %   Find 36x36 faces in an image
# %
# % Inputs:
# %   img: an image
# %   stride: how far to move between locations at which the detector is run
# %   thresh: probability threshold for calling a detection a face
# %   params: trained face classifier parameters
# %   orientations: the number of HoG gradient orientations to use
# %   wrap180: if true, the HoG orientations cover 180 degrees, else 360
# % Outputs:
# %   out_img: copy of img with face locations marked
# %   prob_map: probability map of face detections
# %

def find_faces(img, stride, thresh, params, orientations, wrap180):
    N_img, D_img = img.shape
    out_img = img
    img_resize = np.copy(img)
    N, D = img_resize.shape
    window_size = 36
    stride = min(stride, window_size)
    hog_descriptor_size = 100 * orientations

    while N >= window_size and D >= window_size:
        row_strides = len(range(0, N, stride))
        col_strides = len(range(0, D, stride))
        bar = progressbar.ProgressBar(max_value=row_strides * col_strides, redirect_stdout=True)
        # Loop over window_size x window_size windows, advancing by stride
        k = 0
        for i in range(0, N, stride):
            for j in range(0, D, stride):
                # Crop out a window_size x window_size window starting at (i,j)
                bar.update(k)
                k += 1
                if i + window_size > N or j + window_size > D:
                    continue
                crop = img_resize[i:i+window_size, j:j+window_size]
                # Compute a HoG descriptor, and run the classifier
                descriptor = np.ones(hog_descriptor_size+1)
                descriptor[1:] = hog36(crop, orientations, wrap180)
                probability = logistic_prob(descriptor, params)

                # If probability of a face is below thresh, continue
                if probability < thresh:
                    continue

                # Mark the face in out_img
                out_i = int(np.floor(i * N_img / N))
                out_j = int(np.floor(j * D_img / D))
                out_window_size = int(np.floor(window_size * N_img / N))
                out_img[out_i, out_j:out_j+out_window_size] = 255
                out_img[out_i+out_window_size-1, out_j:out_j+out_window_size] = 255
                out_img[out_i:out_i+out_window_size, out_j] = 255
                out_img[out_i:out_i+out_window_size, out_j+out_window_size-1] = 255

        bar.finish()
        img_resize = imresize(img_resize, 80)
        N, D = img_resize.shape

    return out_img

def find_faces_single_scale(img, stride, thresh, params, orientations, wrap180):
    window_size = 36
    stride = min(stride, window_size)
    hog_descriptor_size = 100 * orientations

    N, D = img.shape
    prob_map = np.zeros((N, D))
    out_img = img

    row_strides = len(range(0, N, stride))
    col_strides = len(range(0, D, stride))
    bar = progressbar.ProgressBar(max_value=row_strides * col_strides, redirect_stdout=True)
    # Loop over window_size x window_size windows, advancing by stride
    k = 0
    for i in range(0, N, stride):
        for j in range(0, D, stride):
            # Crop out a window_size x window_size window starting at (i,j)
            bar.update(k)
            k += 1
            if i + window_size > N or j + window_size > D:
                continue
            crop = img[i:i+window_size, j:j+window_size]
            # Compute a HoG descriptor, and run the classifier
            descriptor = np.ones(hog_descriptor_size+1)
            descriptor[1:] = hog36(crop, orientations, wrap180)
            probability = logistic_prob(descriptor, params)

            # Mark detection probability in prob_map
            win_i = i + int(np.floor((window_size - stride) // 2))
            win_j = j + int(np.floor((window_size - stride) // 2))
            prob_map[win_i:win_i+stride, win_j:win_j+stride] = probability

            # If probability of a face is below thresh, continue
            if probability < thresh:
                continue

            # Mark the face in out_img
            out_img[i, j:j+window_size] = 255
            out_img[i+window_size-1, j:j+window_size] = 255
            out_img[i:i+window_size, j] = 255
            out_img[i:i+window_size, j+window_size-1] = 255

    bar.finish()
    return out_img, prob_map

def find_faces_single_scale_max_supp(img, stride, thresh, params, orientations, wrap180):
    window_size = 36
    stride = min(stride, window_size)
    hog_descriptor_size = 100 * orientations

    N, D = img.shape
    prob_map = np.zeros((N, D))
    out_img = img

    row_strides = len(range(0, N, stride))
    col_strides = len(range(0, D, stride))
    bar = progressbar.ProgressBar(max_value=row_strides * col_strides, redirect_stdout=True)
    # Loop over window_size x window_size windows, advancing by stride
    k = 0
    for i in range(0, N, stride):
        for j in range(0, D, stride):
            # Crop out a window_size x window_size window starting at (i,j)
            bar.update(k)
            k += 1
            if i + window_size > N or j + window_size > D:
                continue
            crop = img[i:i+window_size, j:j+window_size]
            # Compute a HoG descriptor, and run the classifier
            descriptor = np.ones(hog_descriptor_size+1)
            descriptor[1:] = hog36(crop, orientations, wrap180)
            probability = logistic_prob(descriptor, params)

            # Mark detection probability in prob_map
            win_i = i + int(np.floor((window_size - stride) // 2))
            win_j = j + int(np.floor((window_size - stride) // 2))
            prob_map[win_i:win_i+stride, win_j:win_j+stride] = probability

            # If probability of a face is below thresh, continue
            if probability < thresh:
                continue

            # Mark the face in out_img
            out_img[i, j:j+window_size] = 255
            out_img[i+window_size-1, j:j+window_size] = 255
            out_img[i:i+window_size, j] = 255
            out_img[i:i+window_size, j+window_size-1] = 255

    bar.finish()

    plt.imshow(prob_map)
    plt.show()

    def suppress(prob_map, i, j, d):
        if i < 0 or i >= N or j < 0 or j >= N:
            return
        if d == stride // 2:
            return
        cur = prob_map[i][j]
        if cur == 0:
            return
        if cur < prob_map[i-1][j] or \
            cur < prob_map[i+1][j] or \
            cur < prob_map[i+1][j+1] or \
            cur < prob_map[i-1][j-1] or \
            cur < prob_map[i-1][j+1] or \
            cur < prob_map[i+1][j-1] or \
            cur < prob_map[i][j-1] or \
            cur < prob_map[i][j+1]:
            prob_map[i][j] = 0

        suppress(prob_map, i-1, j, d+1)
        suppress(prob_map, i+1, j, d+1)
        suppress(prob_map, i, j-1, d+1)
        suppress(prob_map, i, j+1, d+1)
        suppress(prob_map, i-1, j-1, d+1)
        suppress(prob_map, i+1, j+1, d+1)
        suppress(prob_map, i-1, j+1, d+1)
        suppress(prob_map, i+1, j-1, d+1)

    for i in range(0, N, stride):
        for j in range(0, D, stride):
            if prob_map[i][j] > 0:
                suppress(prob_map, i, j, 0)

    plt.imshow(prob_map)
    plt.show()
    return out_img, prob_map
