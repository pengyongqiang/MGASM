import pickle
from math import exp

import numpy as np
import torch
from smilarity_model import predict_score
from distance_utils import multi_distance


class CustomMetric:

    def __init__(self, model, dim):
        self.model = model
        self.dim = dim

    def __call__(self, x, y):
        # Calculate the distance between x and y using Euclidean distance
        distance = get_distance(x, y, self.dim, self.model)
        return distance


def get_distance(x, y, dim, model):
    dist_array = torch.tensor(np.array([multi_distance(x, y, dim)]), dtype=torch.float32)
    sim_score = predict_score(model, dist_array)[0].item()
    return exp(-sim_score)
