import pickle
import json
import numpy as np
import torch
from scipy.sparse import coo_matrix
from sklearn import metrics
from sklearn.neighbors import BallTree, KDTree
from sklearn.metrics import DistanceMetric
import scipy.sparse as sp
from custom_metric import CustomMetric
from data_utils import find_all_positions

# This function converts the Euclidean distances between the top ten elements closest to network 1 relative to network 2 into a matrix.
def get_sim(emb1, emb2, top_k=10, origin_data=None, granularity_num=None, classifier=None, is_correct=False,
            network_type=None):
    if classifier:
        tree = BallTree(emb2, metric=DistanceMetric.get_metric(CustomMetric(classifier, granularity_num)))
    else:
        tree = KDTree(emb2, metric="euclidean")
    dist, ind = tree.query(emb1, k=top_k)
    row = np.array([])
    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(top_k) * i))

    col = ind.flatten()
    data = np.exp(-dist).flatten()

    sparse_align_matrix = coo_matrix((data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))

    # Whether to use attribute replay correction
    if is_correct:
        if network_type == 'wd':
            coefficient = 1.25
        else:
            coefficient = 1
        attrs = pickle.load(open('../data/' + network_type + '/attrs', 'rb'))
        with open('../data/' + network_type + '/dist.txt', "w", encoding='utf-8') as f:
            for idx, (i, j, v) in enumerate(
                    zip(sparse_align_matrix.row, sparse_align_matrix.col, sparse_align_matrix.data)):
                # Get the original dataset user index corresponding to the generated matrix user index
                true_y = origin_data[i][1]  # Baseline user
                true_x = origin_data[i][0]  # True corresponding
                predict_x = origin_data[j][0]  # Predicted corresponding
                if check_replay(true_y, predict_x, attrs):  # Check for attribute replay
                    dist = sparse_align_matrix.data[idx]
                    sparse_align_matrix.data[idx] = dist * coefficient

                # f.write(f"predict:({i}, {j})==({true_y}, {predict_x})\t{v},true:({true_y}, {true_x})\n")
                # f.write(f"user:{attrs[true_y]}\ntrue:({attrs[true_x]})\npredict:{attrs[predict_x]}\n\n\n")

    return sparse_align_matrix.tocsr()


def get_dimension(array):
    if isinstance(array, list):
        return 1 + get_dimension(array[0])
    else:
        return 0


def calculate_score(sim_matrix, score_type):
    assert sp.issparse(sim_matrix)
    n_nodes = sim_matrix.shape[0]
    nodes = list(range(n_nodes))
    if score_type == 'hit_precision':
        scores = []
        for top_k in [1, 3, 5]:
            score = 0
            for test_x in nodes:
                test_y = test_x
                row_idx, col_idx, values = sp.find(sim_matrix[test_x])
                sorted_idx = col_idx[values.argsort()][-top_k:][::-1]
                h_x = 0
                for pos, idx in enumerate(sorted_idx):
                    if idx == test_y:
                        hit_x = pos + 1
                        h_x = (top_k - hit_x + 1) / top_k
                        break
                score += h_x
            score /= n_nodes
            scores.append(score)
        return scores
    else:
        test = []
        for test_x in nodes:
            if test_x < 250:
                test.append(1)
            else:
                test.append(0)
        row_idx, col_idx, values = sp.find(sim_matrix)
        pred = [0] * n_nodes
        for idx, row in enumerate(row_idx):
            pred[row] = row == col_idx[idx]
        score = metrics.precision_recall_fscore_support(test, pred, average='binary')
        return list(score[:3])


# Check if the username is replayed
def check_replay(x_user, y_user, attrs):
    x_name = attrs[x_user][0]
    y_name = attrs[y_user][0]

    x_topic = attrs[x_user][2]
    y_topic = attrs[y_user][2]

    if len(x_name) > 2 and len(y_name) > 2 and (x_name in y_topic or y_name in x_topic):
        return True
    else:
        return False


def get_user_tag(topic_attrs):
    count = 0
    user_tags = []
    for user_topic in topic_attrs:
        position = find_all_positions(user_topic, '来自')
        need_fill = True
        for start in position:
            if start + 3 < len(user_topic) and user_topic[start + 2] == '[':
                phone_start = start + 3
                phone_end = user_topic.find(']', phone_start)
                if not phone_end == -1:
                    tag = user_topic[phone_start:phone_end]
                    user_tags.append(tag)
                    need_fill = False
                    break
        if need_fill:
            count += 1
            user_tags.append('')
    return user_tags
