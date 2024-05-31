import numpy as np
import json, pickle, os

import pandas as pd
import torch
from tabulate import tabulate

from align_utils import get_sim, calculate_score
from data_utils import calculate_mean
from distance_utils import multi_distance
from smilarity_model import train_model
from data_set import DataSet
from functools import partial


def psearch(train_num, emb_result, indicator_type, correct, dataset, method, random_seed):
    anchors = dict(json.load(open(f'../dataset/{dataset}/anchors.txt', 'r')))
    datasets = DataSet(anchors)

    # Get training and testing sets
    train_set, test_set = datasets.get(n_train=train_num, n_test=500, seed=random_seed)

    # Generate a certain number of non-matching users from the training set randomly
    non_match_train_set = datasets.non_match(anchors, dataset, test_set, train_num, random_seed)

    x_positive, y_positive = [], []
    x_negative, y_negative = [], []
    for k, v in train_set:
        x_positive.append(emb_result[k])
        y_positive.append(emb_result[v])

    for k, v in non_match_train_set:
        x_negative.append(emb_result[k])
        y_negative.append(emb_result[v])

    positive_example = []
    for idx, x in enumerate(x_positive):
        y = y_positive[idx]
        positive_example.append(multi_distance(x, y, dim))

    negative_example = []
    for idx, x in enumerate(x_negative):
        y = y_negative[idx]
        negative_example.append(multi_distance(x, y, dim))

    positive_example = torch.tensor(np.array(positive_example), dtype=torch.float32)
    negative_example = torch.tensor(np.array(negative_example), dtype=torch.float32)

    model_filename = f'../models/sim_model/{dataset}/sm_dim_{dim}_train_{train_num}_{method}.pth'
    # if os.path.exists(model_filename):
    #     sm = torch.load(model_filename)
    #     print("模型已加载")
    # else:
    #     sm = train_model(positive_example, negative_example)
    #     torch.save(sm, model_filename)
    #     print("模型已训练并保存")

    sm = train_model(positive_example, negative_example, dataset)

    x_test, y_test = [], []
    for k, v in test_set:
        x_test.append(emb_result[k])
        y_test.append(emb_result[v])

    if indicator_type == 'hit_precision':
        top_k = 100
    else:
        top_k = 1
        x_test = x_test[:250]
        y_test = y_test[:250]
        non_match_test_set = datasets.non_match(anchors, dataset, train_set, 250, random_seed)
        for k, v in non_match_test_set:
            x_test.append(emb_result[k])
            y_test.append(emb_result[v])

    x_test = torch.tensor(np.array(x_test), dtype=torch.float32)
    y_test = torch.tensor(np.array(y_test), dtype=torch.float32)

    sim_matrix = get_sim(x_test, y_test, top_k, origin_data=test_set, dim=dim,
                         sim_model=sm, is_correct=correct, network_type=dataset)

    return calculate_score(sim_matrix, indicator_type)


if __name__ == '__main__':
    for indicator in ['hit_precision','prf']:
        if indicator == 'hit_precision':
            headers = ['dataset', 'dim', "model", 'n_train', 'AR', 'k=1', 'k=3', 'k=5', 'k=10', 'k=30', 'k=50', 'k=100']
            # headers = ['dataset', 'dim', "model", 'n_train', 'AR', 'k=1', 'k=3', 'k=5']

        else:
            headers = ['dataset', 'dim', "model", 'n_train', 'AR', 'Pre.', 'Rec.', 'F1']

        for dataset in ['wd', 'dblp']:
            scores = []
            # for dim in [50, 100, 150, 200, 250, 300]:
            for dim in [100]:

                for idx, method_name in enumerate(['MGASM', 'MGASM_NL', 'MGASM_LDA']):
                    emb_path = f'../emb/{dataset}/emb_dim_{dim}/'
                    char = np.load(emb_path + "char.npy")
                    word = np.load(emb_path + "word.npy")
                    article_plm = np.load(emb_path + "article_plm.npy")
                    article_lda = np.load(emb_path + "article_lda.npy")
                    struc = np.load(emb_path + "struc.npy")
                    label = np.load(emb_path + "label.npy")
                    MGASM = np.concatenate((char, label, word, article_plm, struc), axis=-1)
                    MGASM_NL = np.concatenate((char, word, article_plm, struc), axis=-1)
                    MGASM_LDA = np.concatenate((char, label, word, article_lda, struc), axis=-1)
                    emb = [MGASM, MGASM_NL, MGASM_LDA][idx]

                    # n_train_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
                    n_train_list = [0.2]
                    for n_train in n_train_list:
                        if dim == 100 and n_train == 0.2:
                            correct_type = [False]
                        else:
                            correct_type = [False]

                        for ar in correct_type:
                            psearch_fun = partial(psearch, n_train, emb, indicator, ar, dataset, method_name)
                            score_10 = []

                            for seed in list(range(5)):
                                print(f"第{seed + 1}/{5}次实验")
                                score_10.append(psearch_fun(seed))
                            score_10 = np.array(score_10)

                            if indicator == 'hit_precision':
                                score = calculate_mean(score_10, 1)
                            else:
                                score = calculate_mean(score_10, 2)

                            score = np.array([round(item, 4) for item in score])
                            record = [dataset, dim, method_name, n_train, ar] + score.tolist()
                            scores.append(record)
            table = tabulate(scores, headers, tablefmt="grid")
            print(table)
            df = pd.DataFrame(scores, columns=headers)
            # df.to_excel(f"../result/{dataset}_{indicator}.xlsx", index=False)
