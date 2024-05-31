import random

import numpy as np


class DataSet(object):
    def __init__(self, anchors):
        assert type(anchors) == type(dict())
        data = np.array(list(anchors.items()))
        self.data = data

    def get(self, n_train, n_test, seed):
        if 0 < n_train < 1: n_train = int(len(self.data) * n_train)
        if 0 < n_test < 1: n_test = int(len(self.data) * n_test)
        np.random.seed(seed)
        data = self.data.copy()
        np.random.shuffle(data)
        return data[:n_train], data[-n_test:]

    def non_match(self, anchors, dataset, test_set, non_size, seed):
        if 0 < non_size < 1: non_size = int(len(self.data) * non_size)
        if dataset == 'dblp': non_size = non_size // 2

        np.random.seed(seed)
        x_list, y_list = test_set[:, -2], test_set[:, -1]
        non_list = []
        while len(non_list) < non_size:
            x = random.choice(x_list)
            y = random.choice(y_list)
            while y == anchors.get(x):
                y = random.choice(y_list)
            non_list.append([x, y])
        return non_list
