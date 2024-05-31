import json
import os
import pickle
import re
from collections import Counter

import jieba
import numpy as np
from opencc import OpenCC
from zhconv import zhconv


# 繁体转简体，并输出为小写
def t2s_and_lower(text):
    cc = OpenCC('t2s')
    return cc.convert(text).lower()


# 分词
def tokenizer_cn(text, pattern=re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9]+')):
    text = t2s_and_lower(text).strip()
    tokens = jieba.lcut(text)
    tokens = [pattern.sub('', token) for token in tokens]
    return tokens


def find_all_positions(string, character):
    positions = []
    index = 0
    while index < len(string):
        position = string.find(character, index)
        if position == -1:
            break
        positions.append(position)
        index = position + 1
    return positions


def check_unanalyzable(words, model):
    for word in words:
        if word not in model.wv:
            return False
    return True


def z_score_normalization(tensor):
    mean = tensor.mean()
    std = tensor.std()
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor


def calculate_mean(data, compare_index=0):
    # 获取指定索引位置的数值
    values = [data_array[compare_index] for data_array in data]

    # 找到指定位置数值最小的两个索引
    min_indices = np.argsort(values)[:2]

    # 移除指定位置数值最小的两组数据
    removed_data = np.delete(data, min_indices, axis=0)

    return np.mean(removed_data, axis=0)
