import pickle
from math import exp

import numpy as np
import torch
from distance_utils import multi_distance
from similarity_model import predict_score

attrs = pickle.load(open('../data/wd/attrs', 'rb'))


class CustomMetric:

    def __init__(self, model, granularity_num):
        self.model = model
        self.granularity_num = granularity_num

    def __call__(self, x, y):
        # Calculate the distance between x and y using Euclidean distance
        distance = get_distance(x, y, self.granularity_num, self.model)
        return distance


def get_distance(x, y, granularity_num, model):
    dist_array = torch.tensor(np.array([multi_distance(x, y, granularity_num)]), dtype=torch.float32)
    sim_score = predict_score(model, dist_array)[0].item()
    return exp(-sim_score)
