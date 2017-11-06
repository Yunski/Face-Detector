import sys
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(X, W):
    return sigmoid(X.dot(W))

def get_data(filename):
    with open(filename) as f:
        content = f.readlines()
        data = np.array([list(map(np.float32, s.split(" "))) for s in content])
        split = np.hsplit(data, np.array([2, 3]))
        return np.array(list(map(np.float32, split[0]))), np.array(list(map(np.int32, np.ndarray.flatten(split[1]))))

def mse(P, Y):
    return np.mean(np.square(P-Y))

def classification_rate(P, Y):
    return np.mean(np.round(P) == Y)

def plot_errors(predicted, classes, is_training):
    nthresh = 99
    npts = predicted.size
    falsepos = np.zeros(nthresh)
    falseneg = np.zeros(nthresh)

    stepsize = 1 / (nthresh + 1)

    for i in range(nthresh):
        thresh = (i+1) * stepsize
        falsepos[i] = np.sum((predicted >= thresh) & (classes == 0)) / npts
        falseneg[i] = np.sum((predicted < thresh) & (classes == 1)) / npts

    limit = 1e-4
    plt.loglog(np.maximum(falsepos, limit), np.maximum(falseneg, limit), 'o')
    data_set = "training" if is_training else "test"
    plt.title('Performance on {} set for varying threshold'.format(data_set))
    plt.xlabel('False positive rate')
    plt.ylabel('False negative rate')
    plt.xlim([limit,1])
    plt.ylim([limit,1])
    plt.show()

def logistic_fit(X_train, Y_train, l2=1e-3, eps=1e-3):
    N, D = X_train.shape
    Y_scale = 2*Y_train-1
    theta = np.linalg.solve(X_train.T.dot(X_train) + l2*N*np.eye(D), X_train.T.dot(Y_scale))
    pY_train = forward(X_train, theta)
    r = Y_train - pY_train
    best_err = sys.maxsize
    errors = []
    i = 0
    while True:
        dS = np.ndarray.flatten(pY_train * (1 - pY_train))
        W = np.diag(dS)
        J = W.dot(X_train)
        dtheta = np.linalg.solve(J.T.dot(J) + l2*N*np.eye(D), J.T.dot(r))
        err = mse(pY_train, Y_train)
        print("error at iter {}: {}".format(i, err))
        errors.append(err)
        if np.abs(best_err - err) < eps:
            break
        if best_err > err:
            best_err = err
        theta += dtheta
        pY_train = forward(X_train, theta)
        r = Y_train - pY_train
        i += 1

    return theta, errors


def logistic_prob(X_test, theta):
    return forward(X_test, theta)

if __name__ == '__main__':
    X_train, Y_train = get_data("training_data.txt")
    X_test, Y_test = get_data("test_data.txt")
    bias = np.ones((len(X_train), 1))
    X_train = np.hstack([bias, X_train])
    X_test = np.hstack([bias, X_test])
    theta, _ = logistic_fit(X_train, Y_train)
    print(theta)
    p_train = logistic_prob(X_train, theta)
    print(classification_rate(p_train, Y_train))
    plot_errors(p_train, Y_train, is_training=True)
    p_test = logistic_prob(X_test, theta)
    print(classification_rate(p_test, Y_test))
    plot_errors(p_test, Y_test, is_training=False)
