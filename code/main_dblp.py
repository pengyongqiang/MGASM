import numpy as np
import json, pickle, os

import pandas as pd
import torch
from tabulate import tabulate

from distance_utils import multi_distance
from smilarity_model import train_model
from data_set import DataSet
from utils import get_sim, calculate_score
from multiprocessing import Pool
from functools import partial


def psearch(n_train, emb, granularity_num, score_type, is_correct, network_type, seed):
    anchors = dict(json.load(open('../data/dblp/anchors.txt', 'r')))
    datasets = DataSet(anchors)

    # Get training and testing sets
    train_set, test_set = datasets.get(n_train=n_train, n_test=500, seed=seed)

    # Generate a certain number of non-matching users from the training set randomly
    non_match_train_set = datasets.non_match(anchors, train_set, n_train//2, seed)

    x_positive, y_positive = [], []
    x_negative, y_negative = [], []
    for k, v in train_set:
        x_positive.append(emb[k])
        y_positive.append(emb[v])

    for k, v in non_match_train_set:
        x_negative.append(emb[k])
        y_negative.append(emb[v])

    positive_example = []
    negative_example = []
    for idx, x in enumerate(x_positive):
        y = y_positive[idx]
        positive_example.append(multi_distance(x, y, granularity_num))

    for idx, x in enumerate(x_negative):
        y = y_negative[idx]
        negative_example.append(multi_distance(x, y, granularity_num))

    positive_example = torch.tensor(np.array(positive_example), dtype=torch.float32)
    negative_example = torch.tensor(np.array(negative_example), dtype=torch.float32)
    # Train a classifier to determine the likelihood that individual-level attributes are correct results under the current similarity
    classifier = train_model(positive_example, negative_example)
    x_test, y_test = [], []
    for k, v in test_set:
        x_test.append(emb[k])
        y_test.append(emb[v])

    if score_type == 'hit_precision':
        top_k = 10
    else:
        top_k = 1
        x_test = x_test[:250]
        y_test = y_test[:250]
        # Generate a certain number of non-matching users from the test set for testing pre., rec., F1 indicators
        non_match_test_set = datasets.non_match(anchors, train_set, 250, seed)
        for k, v in non_match_test_set:
            x_test.append(emb[k])
            y_test.append(emb[v])

    x_test = torch.tensor(np.array(x_test), dtype=torch.float32)
    y_test = torch.tensor(np.array(y_test), dtype=torch.float32)

    sim_matrix = get_sim(x_test, y_test, top_k, origin_data=test_set, granularity_num=granularity_num,
                         classifier=classifier, is_correct=is_correct, network_type=network_type)

    return calculate_score(sim_matrix, score_type)


if __name__ == '__main__':
    print('Model is running..., this process generally takes less than 15 minutes, and the test results will be printed to the console and also saved in the result file.')
    pool = Pool(min(5, os.cpu_count() - 2))
    g1, g2 = pickle.load(open('../data/dblp/networks', 'rb'))
    scores = []
    # for dim in [50, 100, 150, 200]:
    for dim in [100]:
        emb_path = '../emb/emb_dblp_seed_0_dim_{}/emb'.format(dim)
        emb_c, emb_l, emb_w, emb_a_pre, emb_a_lda, emb_g = pickle.load(open(emb_path, 'rb'))
        emb_default = np.concatenate((emb_c, emb_l, emb_w, emb_a_pre, emb_g), axis=-1)
        emb_lda = np.concatenate((emb_c, emb_l, emb_w, emb_a_lda, emb_g), axis=-1)
        emb_no_l = np.concatenate((emb_c, emb_w, emb_a_pre, emb_g), axis=-1)
        # hit_precision: represents calculating hit precision, prf represents calculating Pre., Rec., F1
        # for score_type in ['hit_precision', 'prf']:
        for score_type in ['hit_precision']:
            if score_type == 'hit_precision':
                headers = ['dim', "model_name", 'n_train', 'is_correct', 'k=1', 'k=3', 'k=5']
            else:
                headers = ['dim', "model_name", 'n_train', 'is_correct', 'Pre.', 'Rec.', 'F1']
            # Whether to use attribute correction
            for is_correct in [True, False]:
                # Model variants
                for model in range(3):
                    emb = [emb_default, emb_lda, emb_no_l][model]
                    model_name = ['MGASM', 'MGASM_LDA', 'MGASM_NL'][model]
                    if model_name == 'MGASM_NL':
                        granularity_num = 4
                    else:
                        granularity_num = 5
                    # Training seed number
                    # for n_train in [50,  100, 150,  200, 250]:
                    for n_train in [200]:
                        seed_ = list(range(1))
                        psearch_fun = partial(psearch, n_train, emb, granularity_num, score_type, is_correct, 'dblp')
                        score_10 = pool.map(psearch_fun, seed_)  # Multi-threaded way to call
                        score_10 = np.array(score_10)
                        score = np.mean(score_10, axis=0)
                        score = np.array([round(item, 4) for item in score])
                        record = [dim, model_name, n_train, is_correct] + score.tolist()
                        scores.append(record)
                        print(record)
            table = tabulate(scores, headers, tablefmt="grid")
            print(table)
            # Save as an Excel table
            df = pd.DataFrame(scores, columns=headers)
            # df.to_excel("../result/MGASM_DBLP_{}.xlsx".format(score_type), index=False)
