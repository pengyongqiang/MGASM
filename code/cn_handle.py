import copy
import pickle
import re

import jieba
import zhconv
from pypinyin import lazy_pinyin

from data_utils import find_all_positions


# Chinese tokenization
def tokenizer_cn(text):
    zh_matcher = re.compile('[^\u4e00-\u9fa5]')
    # Convert traditional Chinese characters to simplified
    text = zhconv.convert(text, 'zh-hans').strip()  # Standardize to simple Chinese
    # Replace non-Chinese characters with empty string
    text = zh_matcher.sub('', text)
    return jieba.lcut(text)


# Chinese character processing -- remove special symbols, digits, spaces, and split by character
def char_cn(char_attrs):
    # Convert Chinese characters to Pinyin, then to lowercase, and remove punctuation, numbers, and other characters
    punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    char_docs = []
    for doc in char_attrs:
        pinyin = list(''.join(lazy_pinyin(doc)).lower())
        char_docs.append([c for c in pinyin if not c.isnumeric() and not c.isspace() and not c in punc])
    return char_docs


# Chinese word preprocessing
def word_cn(docs):
    stop_words = set(pickle.load(open('../data/wd/stop_words_cn.pkl', 'rb')))
    docs = [[word for word in document if word not in stop_words] for document in docs]
    return docs


# Chinese article preprocessing
def article_cn_pre(attrs):
    docs = copy.deepcopy(attrs)
    for idx, user_article in enumerate(docs):
        # Remove label granularity attributes that may affect the results
        user_article = re.sub(r'来自\[[^\]]+\]', '', user_article)
        user_article = re.sub(r'https?://\S+|www\.\S+', '', user_article)
        user_article = re.sub(r'(赞|转发|评论)\[\d+\]', '', user_article)
        docs[idx] = user_article
    return docs


def article_cn_lda(attrs):
    docs = copy.deepcopy(attrs)
    stop_words = pickle.load(open('../data/wd/stop_words_cn.pkl', 'rb'))
    stop_words = set(stop_words)
    for idx, user_article in enumerate(docs):
        docs[idx] = [token for token in tokenizer_cn(user_article) if token not in stop_words]
    return docs
