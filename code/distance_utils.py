import numpy as np


def euclidean_distance(arr1, arr2):
    return np.sqrt(np.sum((arr1 - arr2) ** 2))


def multi_distance(x, y, num_splits):
    dists = []
    x = np.array_split(x, num_splits)
    y = np.array_split(y, num_splits)
    for idx, gran_x in enumerate(x):
        gran_y = y[idx]
        dis = euclidean_distance(gran_x, gran_y)
        if dis == 0:
            dis = 1e-13
        dists = np.concatenate((dists, [dis]))
    return dists
