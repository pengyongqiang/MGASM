import copy
import os
import re
import time

import jieba
import networkx as nx
import numpy as np
from gensim import models
from gensim.corpora import WikiCorpus, Dictionary
from gensim.models.word2vec import LineSentence, Word2Vec
from nltk import WordNetLemmatizer, word_tokenize
from node2vec import Node2Vec
from pypinyin import lazy_pinyin

from autoencoder import AE
from data_utils import z_score_normalization, t2s_and_lower, check_unanalyzable
from data_process import en_preproc, cn_preproc
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import ElectraTokenizer, ElectraForSequenceClassification, BertForSequenceClassification, \
    BertTokenizer
import path_conf
from nltk.corpus import stopwords


def char_embedding(users, attr_name, dim, lan):
    # 定义标点符号集合，提高检查效率
    punc_set = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    if lan == 'cn':
        docs = cn_preproc(users, attr_name)
    else:
        docs = en_preproc(users, attr_name)

    for idx, doc in enumerate(docs):
        attr_value = t2s_and_lower(' '.join(doc))
        char_list = [c for c in attr_value if not (c.isnumeric() or c.isspace() or c in punc_set)]
        docs[idx] = char_list

    # Build tri-gram phrases
    bi_gram = models.Phrases(docs, min_count=5, threshold=0.1)
    tri_gram = models.Phrases(bi_gram[docs], min_count=5, threshold=0.1)

    # Build Phraser
    bi_phraser = models.phrases.Phraser(bi_gram)
    tri_phraser = models.phrases.Phraser(tri_gram)

    # Merge documents after phrases
    tri_phraser_docs = [tri_phraser[bi_phraser[doc]] for doc in docs]

    # Generate dictionary
    tri_dict = Dictionary(tri_phraser_docs)
    _ = tri_dict[0]

    # Get bag-of-words representation of vectors
    bow_docs = [tri_dict.doc2bow(doc) for doc in tri_phraser_docs]

    # Convert bag-of-words to term frequency vectors
    data = np.zeros((len(bow_docs), len(tri_dict)))

    for n, values in enumerate(bow_docs):
        for idx, value in values:
            data[n][idx] = value

    embed_result = np.array(AE(emb_dim=dim, data=data, epochs=100))

    for idx, user in enumerate(users):
        attr = getattr(user, attr_name)
        if not attr.strip():
            embed_result[idx] = np.full(dim, 1e-13, dtype=np.float64)

    return z_score_normalization(embed_result)


# def char_embedding(users, attr_name, dim):
#     # 定义标点符号集合，提高检查效率
#     punc_set = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
#     docs = []  # 最终存储处理结果的列表
#     for user in users:
#         attr_value = getattr(user, attr_name)
#         attr_value = t2s_and_lower(attr_value)
#         char_list = [w for w in attr_value if not (w.isnumeric() or w.isspace() or w in punc_set)]
#         docs.append(char_list)
#
#     # Build tri-gram phrases
#     bi_gram = models.Phrases(docs, min_count=5, threshold=0.1)
#     tri_gram = models.Phrases(bi_gram[docs], min_count=5, threshold=0.1)
#
#     # Build Phraser
#     bi_phraser = models.phrases.Phraser(bi_gram)
#     tri_phraser = models.phrases.Phraser(tri_gram)
#
#     # Merge documents after phrases
#     tri_phraser_docs = [tri_phraser[bi_phraser[doc]] for doc in docs]
#
#     # Generate dictionary
#     tri_dict = Dictionary(tri_phraser_docs)
#     _ = tri_dict[0]
#
#     # Get bag-of-words representation of vectors
#     bow_docs = [tri_dict.doc2bow(doc) for doc in tri_phraser_docs]
#
#     # Convert bag-of-words to term frequency vectors
#     data = np.zeros((len(bow_docs), len(tri_dict)))
#
#     for n, values in enumerate(bow_docs):
#         for idx, value in values:
#             data[n][idx] = value
#
#     embed_result = np.array(AE(emb_dim=dim, data=data, epochs=50))
#
#     for idx, user in enumerate(users):
#         attr = getattr(user, attr_name)
#         if not attr.strip():
#             embed_result[idx] = np.full(dim, 1e-13, dtype=np.float64)
#
#     return z_score_normalization(embed_result)


def word_embedding(users, attr_name, dim, g1, g2, lan):
    if lan == "cn":
        path_prefix = path_conf.cn
        docs = cn_preproc(users, attr_name)
        # 构建维基语料库(如果之前构建过了就不再构建了)
        if not os.path.exists(path_prefix.out_corpus_file) or os.path.getsize(path_prefix.out_corpus_file) == 0:
            output = open(path_prefix.out_corpus_file, 'w', encoding='utf-8')
            wiki = WikiCorpus(path_prefix.input_corpus_file, processes=10, lemmatize=False, lower=False)
            for idx, words in enumerate(wiki.get_texts()):
                words = [' '.join(jieba.lcut(w)) for w in words]
                output.write(' '.join(words) + '\n')
                if (idx + 1) % 10000 == 0: print('已完成:{}条'.format(idx + 1))
            output.close()

        model_path = path_prefix.wv_save_file + f'word2vec{dim}.model'
        iter = LineSentence(path_prefix.out_corpus_file)
    else:
        path_prefix = path_conf.en
        docs = en_preproc(users, attr_name)

        model_path = path_prefix.wv_save_file + f'word2vec{dim}.model'
        iter = docs

    if not os.path.exists(model_path):
        model = Word2Vec(sentences=iter, size=dim, workers=os.cpu_count() - 1)
        model.save(model_path)
    else:
        model = Word2Vec.load(model_path)

    embed_result = []  # 存储嵌入结果

    for idx, doc in enumerate(docs):
        sentence_vector = []
        for word in doc:
            if word in model.wv:
                sentence_vector.append(model.wv[word])
        if len(sentence_vector) == 0:  # 如果没有嵌入结果则填充极小值
            embed_result.append(np.full(dim, 1e-13, dtype=np.float64))
        else:
            embed_result.append(np.mean(sentence_vector, axis=0))
    embed_result = np.array(embed_result)

    if lan == 'cn':
        embed_copy = embed_result.copy()
        for idx, doc in enumerate(docs):
            if len(doc) == 0:
                nei_emb = np.full(dim, 1e-13, dtype=np.float64)
                count = 0
                if idx < len(g1):
                    for j in g1.neighbors(idx):
                        nei_emb += embed_copy[j]
                        count += 1
                else:
                    for j in g2.neighbors(idx):
                        nei_emb += embed_copy[j]
                        count += 1
                embed_result[idx] = nei_emb * (1 / count)
        return z_score_normalization(embed_copy)
    else:
        return z_score_normalization(embed_result)


def lda_embedding(users, dim, lan):
    if lan == 'cn':
        articles = cn_preproc(users, 'article')
    else:
        articles = en_preproc(users, 'article')

    docs_dict = Dictionary(articles)

    # 过滤掉在所有文档中出现频率小于10的词语，或者在50%以上文档中都出现的词
    docs_dict.filter_extremes(no_below=10, no_above=0.5)

    # 构建词袋语料库
    corpus = [docs_dict.doc2bow(doc) for doc in articles]

    # 这里可能是bug，必须要调用一下dictionary才能确保被加载
    _ = docs_dict[0]

    # 得到字典中id到词语的映射集合
    id2word = docs_dict.id2token

    model = models.LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=256,
        passes=1,
        iterations=100,
        num_topics=dim,
        eval_every=1,
        minimum_probability=1e-13,
        random_state=0)
    article_distribution = model.get_document_topics(corpus)

    embed_result = []
    for i in range(len(corpus)):
        emb = []
        article_i = article_distribution[i]
        # 如果emb的维度低于dimension,则说明部分文章维度太小被省略了，则需要进行填充
        if len(article_i) < dim:
            article_i = dict(article_i)
            for j in range(dim):
                if j in article_i.keys():
                    emb.append(article_i[j])
                else:
                    emb.append(1e-13)
        else:
            article_i = article_i[0:dim]
            emb = np.array(article_i, dtype=np.float64)[:, 1]
        embed_result.append(emb)

    embed_result = np.array(embed_result)
    return z_score_normalization(embed_result)


def graph_embed(g, dim):
    if not nx.is_directed(g):
        g = g.to_directed()
    g = nx.relabel_nodes(g, {node: str(node) for node in g.nodes})

    node2vec = Node2Vec(g, dimensions=dim, walk_length=10, num_walks=5)

    model = node2vec.fit(window=10, min_count=1, batch_words=4, workers=4)

    embed_result = {str(node): model.wv[str(node)] for node in g.nodes}

    return embed_result


def electra_embed(users, batch_size, dim):
    docs = []
    for idx, user in enumerate(users):
        user_article = getattr(user, 'article')
        user_article = re.sub(r'来自\[[^\]]+\]', '', user_article)
        user_article = re.sub(r'https?://\S+|www\.\S+', '', user_article)
        user_article = re.sub(r'(赞|转发|评论)\[\d+\]', '', user_article)
        docs.append(user_article)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ElectraForSequenceClassification.from_pretrained(path_conf.cn.chinese_electra, num_labels=dim).to(device)
    tokenizer = ElectraTokenizer.from_pretrained(path_conf.cn.chinese_electra)
    encoded_input = tokenizer(docs, add_special_tokens=True, padding=True, max_length=256,
                              truncation=True, return_tensors='pt').to(device)

    dataset = TensorDataset(encoded_input['input_ids'].to(device), encoded_input['attention_mask'].to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size)
    embed = None
    progress_bar = tqdm(total=len(docs))
    with torch.no_grad():
        for batch in dataloader:
            batch_input_ids = batch[0].to(device)
            batch_attention_mask = batch[1].to(device)
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            probas = torch.nn.functional.softmax(outputs[0], dim=-1)
            if embed is None:
                embed = probas
            else:
                embed = torch.cat((embed, probas), dim=0)
            progress_bar.update(batch_size)

    return z_score_normalization(embed.cpu().detach())


def electra_embed_en(users, batch_size, dim):
    docs = []
    for idx, user in enumerate(users):
        user_article = getattr(user, 'article')
        docs.append(user_article)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ElectraForSequenceClassification.from_pretrained(path_conf.en.english_electra, num_labels=dim).to(device)
    tokenizer = ElectraTokenizer.from_pretrained(path_conf.en.english_electra)
    encoded_input = tokenizer(docs, add_special_tokens=True, padding=True, max_length=256,
                              truncation=True, return_tensors='pt').to(device)

    dataset = TensorDataset(encoded_input['input_ids'].to(device), encoded_input['attention_mask'].to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size)
    embed = None
    progress_bar = tqdm(total=len(docs))
    with torch.no_grad():
        for batch in dataloader:
            batch_input_ids = batch[0].to(device)
            batch_attention_mask = batch[1].to(device)
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            probas = torch.nn.functional.softmax(outputs[0], dim=-1)
            if embed is None:
                embed = probas
            else:
                embed = torch.cat((embed, probas), dim=0)
            progress_bar.update(batch_size)

    return z_score_normalization(embed.cpu().detach())



# def bert_embed(users, batch_size, dim):
#     docs = []
#     for idx, user in enumerate(users):
#         user_article = getattr(user, 'article')
#         docs.append(user_article)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = BertForSequenceClassification.from_pretrained(path_conf.en.bert_base_ner, num_labels=dim,
#                                                           ignore_mismatched_sizes=True).to(device)
#     tokenizer = BertTokenizer.from_pretrained(path_conf.en.bert_base_ner)
#     encoded_input = tokenizer(docs, add_special_tokens=True, padding=True, max_length=256,
#                               truncation=True, return_tensors='pt').to(device)
#
#     dataset = TensorDataset(encoded_input['input_ids'].to(device), encoded_input['attention_mask'].to(device))
#
#     # Create DataLoader
#     dataloader = DataLoader(dataset, batch_size=batch_size)
#
#     embed = None
#     progress_bar = tqdm(total=len(docs))
#     # Process each batch
#     with torch.no_grad():
#         for batch in dataloader:
#             batch_input_ids = batch[0].to(device)
#             batch_attention_mask = batch[1].to(device)
#             outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
#             probas = torch.nn.functional.softmax(outputs[0], dim=-1)
#             if embed is None:
#                 embed = probas
#             else:
#                 embed = torch.cat((embed, probas), dim=0)
#             progress_bar.update(batch_size)
#     return z_score_normalization(embed.cpu().detach())
