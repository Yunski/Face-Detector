import os
import glob
import time
import progressbar
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from hog36 import hog36

def get_training_data(n, orientations, wrap180=False):
    training_faces_dir = '../cos429_f17_assignment2_part2/face_data/training_faces'
    training_nonfaces_dir = '../cos429_f17_assignment2_part2/face_data/training_nonfaces'
    window_size = 36
    hog_descriptor_size = 100 * orientations
    # Get the names of the first n training faces
    face_filenames = glob.glob(training_faces_dir + '/*.jpg')
    num_face_filenames = len(face_filenames)
    if num_face_filenames > n:
        face_filenames = face_filenames[:n]
    elif num_face_filenames <= n:
        n = num_face_filenames
    # Initialize descriptors, classes
    bar = progressbar.ProgressBar(max_value=2*n, redirect_stdout=True)
    descriptors = np.zeros((2*n, hog_descriptor_size))
    bias = np.ones((descriptors.shape[0], 1))
    descriptors = np.hstack([bias, descriptors])
    classes = np.zeros(2*n)

    # Loop over faces
    for i in range(n):
        # Read the next face file
        path = os.path.abspath(face_filenames[i])
        img = imread(path, mode='L')
        face_descriptor = hog36(img, orientations, wrap180)
        # Fill in descriptors and classes
        descriptors[i, 1:hog_descriptor_size+1] = face_descriptor
        bar.update(i)
    classes[:n] = 1

    # Get the names of the nonfaces
    nonface_filenames = glob.glob(training_nonfaces_dir + '/*.jpg')
    num_nonface_filenames = len(nonface_filenames)

    # % Loop over all nonface samples we want
    for i in range(n, 2*n):
        # Read a random nonface file
        j = np.random.choice(np.arange(num_nonface_filenames))
        path = os.path.abspath(nonface_filenames[j])
        img = imread(path, mode='L')
        N, D = img.shape
        x = np.random.choice(np.arange(N-window_size))
        y = np.random.choice(np.arange(D-window_size))
        # Crop out a random square at least window_size

        crop = img[x:x+window_size, y:y+window_size]
        # Compute descriptor, and fill in descriptors and classes
        nonface_descriptor = hog36(crop, orientations, wrap180)
        descriptors[i, 1:hog_descriptor_size+1] = nonface_descriptor
        bar.update(i)
    bar.finish()
    return descriptors, classes


def get_testing_data(n, orientations, wrap180=False):
    testing_faces_dir = '../cos429_f17_assignment2_part2/face_data/testing_faces'
    testing_nonfaces_dir = '../cos429_f17_assignment2_part2/face_data/testing_nonfaces'
    window_size = 36
    hog_descriptor_size = 100 * orientations

    # Get the names of the first n training faces
    face_filenames = glob.glob(testing_faces_dir + '/*.jpg')
    num_face_filenames = len(face_filenames)

    if num_face_filenames > n:
        face_filenames = face_filenames[:n]
    elif num_face_filenames <= n:
        n = num_face_filenames

    bar = progressbar.ProgressBar(max_value=2*n, redirect_stdout=True)
    # Initialize descriptors, classes
    descriptors = np.zeros((2*n, hog_descriptor_size))
    bias = np.ones((descriptors.shape[0], 1))
    descriptors = np.hstack([bias, descriptors])
    classes = np.zeros(2*n)

    # Loop over faces
    for i in range(n):
        # Read the next face file
        path = os.path.abspath(face_filenames[i])
        img = imread(path, mode='L')
        face_descriptor = hog36(img, orientations, wrap180)
        # Fill in descriptors and classes
        descriptors[i, 1:hog_descriptor_size+1] = face_descriptor
        bar.update(i)
    classes[:n] = 1

    # Get the names of the nonfaces
    nonface_filenames = glob.glob(testing_nonfaces_dir + '/*.jpg')
    num_nonface_filenames = len(nonface_filenames)

    # % Loop over all nonface samples we want
    for i in range(n, 2*n):
        # Read a random nonface file
        j = np.random.choice(np.arange(num_nonface_filenames))
        path = os.path.abspath(nonface_filenames[j])
        img = imread(path, mode='L')
        nonface_descriptor = hog36(img, orientations, wrap180)
        descriptors[i, 1:hog_descriptor_size+1] = nonface_descriptor
        bar.update(i)
    bar.finish()
    return descriptors, classes
