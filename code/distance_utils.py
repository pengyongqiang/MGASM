import numpy as np


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))


def chebyshev_distance(x, y):
    return np.max(np.abs(x - y))


def cosine_similarity(x, y):
    dot_product = np.dot(x, y)
    norm_vec1 = np.linalg.norm(x)
    norm_vec2 = np.linalg.norm(y)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return 1 - similarity


def hamming_distance(vector1, vector2):
    distance = 0
    for v1, v2 in zip(vector1, vector2):
        if v1 != v2:
            distance += 1

    return distance


def multi_distance(x, y, dim):
    num_splits = x.size // dim
    dists = []
    x = np.array_split(x, num_splits)
    y = np.array_split(y, num_splits)
    for idx, gran_x in enumerate(x):
        gran_y = y[idx]
        dis = cosine_similarity(gran_x, gran_y)
        if dis == 0:
            dis = 1e-16
        dists = np.concatenate((dists, [dis]))
    return dists
