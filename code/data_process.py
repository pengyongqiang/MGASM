import json
import pickle
import re
from collections import Counter

from nltk.corpus import stopwords

from data_utils import tokenizer_cn
from gensim import utils


# 文章处理
def cn_preproc(user_list, attr_name):
    pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9]+')
    docs = []
    for user in user_list:
        attr_value = getattr(user, attr_name)
        attr_value = re.sub(r'https?://\S+|www\.\S+', '', attr_value)
        attr_value = re.sub(r'(赞|转发|评论)\[\d+\]', '', attr_value)
        attr_value = re.sub(r'\[组图共\d+张\]', '', attr_value)
        doc = tokenizer_cn(attr_value, pattern)
        docs.append(doc)

    stop_words = set(pickle.load(open('../models/cn/stop_words_cn.pkl', 'rb')))
    stop_words.update(stopwords.words('chinese'))
    # stop_words.update(['mcn', '置顶', '[置顶]', '转发了', '转发', '现居', '未知', '其他'])
    stop_words.update(['mcn', '置顶', '[置顶]', '转发了', '转发', '现居'])
    docs = [[word for word in doc if word not in stop_words] for doc in docs]

    return docs


def en_preproc(user_list, attr_name, min_len=2, max_len=15):
    stop_words = set(stopwords.words('english'))
    docs = []
    for user in user_list:  # 遍历 user_list 中的每个 user 对象
        docs.append([token for token in
                     utils.tokenize(getattr(user, attr_name), lower=True, deacc=True, errors='ignore')
                     if min_len <= len(token) <= max_len and token not in stop_words])

    # docs = [[word for word in document if word not in stop_words] for document in docs]
    return docs


def get_high_frequency_words(times):
    attrs = pickle.load(open('../dataset/dblp/attrs', 'rb'))
    anchors = dict(json.load(open('../dataset/dblp/anchors.txt', 'r')))
    all_words = []
    stop_words = set(stopwords.words('english'))
    for x in anchors:
        x_words = [token for token in utils.tokenize(attrs[x][1], lower=True, deacc=True, errors='ignore') if
                   token not in stop_words]
        all_words.extend(x_words)

        y_words = [token for token in utils.tokenize(attrs[anchors.get(x)][1], lower=True, deacc=True, errors='ignore')
                   if
                   token not in stop_words]
        all_words.extend(y_words)

    counter = Counter(all_words)
    sorted_counts = counter.most_common()
    filtered_elements = [elem for elem, count in sorted_counts if count > times]
    return filtered_elements
