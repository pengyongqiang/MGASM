import copy
import pickle
import re

import nltk
from gensim import utils, models
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preproc_en(attrs, min_token_len=2, max_token_len=15):
    docs = copy.deepcopy(attrs)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english') + ['from', 'subject', 're', 'edu', 'use'])

    for idx, doc in enumerate(docs):
        new_tokens = []
        for token in word_tokenize(doc.lower()):
            token = lemmatizer.lemmatize(token)
            if min_token_len <= len(token) <= max_token_len and token not in stop_words:
                new_tokens.append(token)
        docs[idx] = new_tokens

    # Build the bi-gram and trigram models
    bi_gram_model = Phrases(docs, min_count=5, threshold=0.1)
    trigram_model = Phrases(bi_gram_model[docs], threshold=0.1)

    # Get a sentence clubbed as a trigram/bi_gram
    bi_gram_phraser = Phraser(bi_gram_model)
    trigram_phraser = Phraser(trigram_model)

    # Add bigrams and trigrams to docs
    preprocessed_docs = [trigram_phraser[bi_gram_phraser[doc]] for doc in docs]

    return preprocessed_docs


def char_en(char_attrs):
    punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    char_attrs = [[w for w in list(doc) if
                   not w.isnumeric()
                   and not w.isspace()
                   and not w in punc] for doc in char_attrs]
    return char_attrs


def label_en(attrs):
    docs = copy.deepcopy(attrs)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english') + ['from', 'subject', 're', 'edu', 'use', 'of', 'university'])
    for idx, doc in enumerate(docs):
        new_tokens = []
        for token in word_tokenize(doc.lower()):
            token = lemmatizer.lemmatize(token)
            if token not in stop_words:
                new_tokens.append(token)
        docs[idx] = new_tokens
    return char_en(docs)
