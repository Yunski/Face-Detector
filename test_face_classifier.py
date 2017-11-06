import argparse
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from data import get_training_data, get_testing_data
from logistic_regression import classification_rate, logistic_fit, logistic_prob, plot_errors

def train_face_classifier(ntrain, ntest, orientations, wrap180, model_save_file):
    print("Loading training data...")
    descriptors_train, classes_train = get_training_data(ntrain, orientations, wrap180=wrap180)
    print("Finished loading training data.")
    print("Loading test data...")
    descriptors_test, classes_test = get_testing_data(ntest, orientations, wrap180=wrap180)
    print("Finished loading test data.")
    print("Start training...")

    start_time = time.time()
    params, _ = logistic_fit(descriptors_train, classes_train)
    print("Training took {} seconds.".format(time.time()-start_time))

    np.save(model_save_file, params)

    predicted_train = logistic_prob(descriptors_train, params)
    plot_errors(predicted_train, classes_train, is_training=True)

    train_success_rate = classification_rate(predicted_train, classes_train)
    print("Training classification rate: {}".format(train_success_rate))

    predicted_test = logistic_prob(descriptors_test, params)
    plot_errors(predicted_test, classes_test, is_training=False)

    test_success_rate = classification_rate(predicted_test, classes_test)
    print("Testing classification rate: {}".format(test_success_rate))


def test_face_classifier(ntrain, ntest, orientations, wrap180, model_save_file):
    print("Loading test data...")
    descriptors_test, classes_test = get_testing_data(ntest, orientations, wrap180=wrap180)
    print("Finished loading test data.")

    params = np.load(model_save_file)
    predicted_test = logistic_prob(descriptors_test, params)
    plot_errors(predicted_test, classes_test, is_training=False)

    test_success_rate = classification_rate(predicted_test, classes_test)
    print("Testing classification rate: {}".format(test_success_rate))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Classifier')
    parser.add_argument('-n', help='number of training samples', dest='ntrain', type=int, default=6000)
    parser.add_argument('-m', help='number of test samples', dest='ntest', type=int, default=500)
    parser.add_argument('-o', help='number of orientations', dest='orientations', type=int, default=9)
    parser.add_argument('-w', help='wrap 180', dest='wrap180', type=int, default=0)
    parser.add_argument('-t', help='is training', dest='train', type=int, default=0)
    parser.add_argument('model_save_file', help='model save file', type=str)
    args = parser.parse_args()
    if args.train:
        train_face_classifier(args.ntrain, args.ntest, args.orientations, args.wrap180, args.model_save_file)
    else:
        test_face_classifier(args.ntrain, args.ntest, args.orientations, args.wrap180, args.model_save_file)
