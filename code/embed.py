import os
import pickle
import time
import networkx as nx
import numpy as np
from gensim import models
from gensim.models.word2vec import LineSentence, Word2Vec
from gensim.corpora import WikiCorpus, Dictionary
from node2vec import Node2Vec
from tensorflow.contrib.layers import fully_connected
import path_conf
from cn_handle import char_cn, word_cn, article_cn_pre, article_cn_lda
from bert_uncased_ner import bert_uncased_ner
from electra import electra_embed
from utils import get_user_tag
from cn_handle import tokenizer_cn
import tensorflow as tf


# Filter out specific types of warnings
def word_embd(word_attrs, dim, g1, g2, emb_path, path_class):
    print(time.ctime(), 'Word granularity attribute embedding begins...')
    if path_conf.wd == path_class:  # If it's Weibo-Douban, a Chinese corpus needs to be built first
        # Build a Chinese corpus from Wikipedia (if not built previously)
        if not os.path.exists(path_class.out_corpus_file) or os.path.getsize(path_class.out_corpus_file) == 0:
            print('Building a Chinese corpus from the local file {}, corpus will be saved at:{}'.format(
                path_class.input_corpus_file, path_class.out_corpus_file))
            output = open(path_class.out_corpus_file, 'w', encoding='utf-8')
            wiki = WikiCorpus(path_class.input_corpus_file, processes=10, lemmatize=False,
                              lower=False)
            count = 0
            # Note that the return value of wiki.get_texts() is an iterator, and each iteration yields a list of words
            for words in wiki.get_texts():
                words = [' '.join(tokenizer_cn(w)) for w in words]
                output.write(' '.join(words) + '\n')
                count += 1
                if count % 10000 == 0:
                    print('Completed:{} lines'.format(count))
            output.close()

        print(time.ctime(), 'Training word vectors using Word2Vec...')
        iter_ = LineSentence(path_class.out_corpus_file)
        model = Word2Vec(sentences=iter_, size=dim, workers=os.cpu_count() - 1)
        model.save(path_class.wv_save_file)

    else:
        print(time.ctime(), 'Training word vectors using Word2Vec...')
        model = Word2Vec(sentences=word_attrs, size=dim, workers=os.cpu_count() - 1)

    embed = []
    for doc in word_attrs:
        emb = np.full(dim, 1e-13, dtype=np.float64)
        for word in doc:
            if word in model.wv:
                emb += model.wv[word]
        embed.append(emb)
    embed = np.array(embed)

    # Smooth the results of word embedding by introducing neighboring node's word granularity attributes
    embed_copy = embed.copy()
    for i in range(len(word_attrs)):
        word = word_attrs[i]
        if not word:
            nei_emb = np.full(dim, 1e-13, dtype=np.float64)
            count = 0
            if i < len(g1):
                for j in g1.neighbors(i):
                    nei_emb += embed_copy[j]
                    count += 1
            else:
                for j in g2.neighbors(i):
                    nei_emb += embed_copy[j]
                    count += 1
            embed[i] = nei_emb * (1 / count)

    print(time.ctime(), 'Word granularity vector embedding completed, shape:{}'.format(embed.shape))

    pickle.dump(embed, open(emb_path + 'emb_w', 'wb'))
    return embed


def article_embed_PRE(docs, dim, emb_path, path_class):
    print(time.ctime(), 'Article granularity attributes embedding using pre-trained model begins...')
    if path_conf.wd == path_class:  # If it's Weibo-Douban, a Chinese corpus needs to be built first
        embed = electra_embed(docs, 1024, dim).numpy()
    else:
        embed = bert_uncased_ner(docs, 512, dim).numpy()

    # Save embedding results
    pickle.dump(embed, open(emb_path + 'emb_a_pre', 'wb'))
    print(time.ctime(), 'Article granularity attributes embedding using pre-trained model completed!')
    return embed


def article_embed_LDA(docs, dim, emb_path):
    print(time.ctime(), 'Article granularity attributes embedding using LDA begins...')

    docs_dict = Dictionary(docs)

    # Filter out words that appear less than 10 times in all documents, or appear in more than 50% of the documents
    docs_dict.filter_extremes(no_below=10, no_above=0.5)

    # Build a bag-of-words corpus
    corpus = [docs_dict.doc2bow(doc) for doc in docs]

    # This may be a bug, you must call dictionary to ensure it is loaded, asked gpt also answered like this
    _ = docs_dict[0]

    # Get the mapping from id to word in the dictionary
    id2word = docs_dict.id2token

    print(time.ctime(), 'Training using LDA topic model...')
    model = models.LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=256,
        passes=4,
        iterations=100,
        num_topics=dim,
        eval_every=1,
        minimum_probability=1e-13,
        random_state=0)

    article_distribution = model.get_document_topics(corpus)

    embed = []
    for i in range(len(corpus)):
        emb = []
        article_i = article_distribution[i]
        # If the dimension of emb is less than dimension, some article dimensions are too small and are omitted, so filling is required
        if len(article_i) < dim:
            article_i = dict(article_i)
            for j in range(dim):
                if j in article_i.keys():
                    emb.append(article_i[j])
                else:
                    emb.append(1e-13)
        else:
            article_i = article_i[0:100]
            emb = np.array(article_i, dtype=np.float64)[:, 1]
        embed.append(emb)
    embed = np.array(embed)
    # Save embedding results
    pickle.dump(embed, open(emb_path + 'emb_a_lda', 'wb'))
    print(time.ctime(), 'Article granularity attributes embedding using LDA completed!')
    return embed


def graph_embed(g, dim, g_name, emb_path):
    print(time.ctime(), 'Embedding of user relationship graph {} begins...'.format(g_name))
    if not nx.is_directed(g):
        g = g.to_directed()
    g = nx.relabel_nodes(g, {node: str(node) for node in g.nodes})

    # Use node2vec for embedding
    node2vec = Node2Vec(g, dimensions=dim, walk_length=10, num_walks=5)

    model = node2vec.fit(window=10, min_count=1, batch_words=4, workers=4)

    # Get embedding vectors for each node
    embeddings = {str(node): model.wv[str(node)] for node in g.nodes}

    # Save embedding vectors to a file
    pickle.dump(embeddings, open(emb_path + 'emb_' + g_name, 'wb'))
    print(time.ctime(), 'Embedding of user relationship graph {} completed!'.format(g_name))


def char_embd(char_docs, dim, emb_path, level):
    emb_name = 'Character'
    if level == 'emb_l': emb_name = 'Label'
    print(time.ctime(), 'Embedding of {} granularity attributes begins...'.format(emb_name))

    # Build tri-gram phrases
    bi_gram = models.Phrases(char_docs, min_count=5, threshold=0.1)
    tri_gram = models.Phrases(bi_gram[char_docs], min_count=5, threshold=0.1)

    # Build Phraser
    bi_phraser = models.phrases.Phraser(bi_gram)
    tri_phraser = models.phrases.Phraser(tri_gram)

    # Merge documents after phrases
    tri_phraser_docs = [tri_phraser[bi_phraser[doc]] for doc in char_docs]

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

    # Train vectors using an autoencoder through two preceding layers
    g = tf.get_default_graph()
    x = tf.placeholder(tf.float64, shape=[None, data.shape[1]])
    hidden = fully_connected(x, dim, activation_fn=None)
    outputs = fully_connected(hidden, data.shape[1], activation_fn=None)

    loss = tf.reduce_mean(tf.square(outputs - x))
    train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

    with tf.Session(graph=g, config=tf.ConfigProto()) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(50):
            _, embed, loss_v = sess.run([train_op, hidden, loss], feed_dict={x: data})
            if epoch % 10 == 0:
                print('Epoch {}/{}, Loss:{}'.format(epoch, 50, loss_v))

    pickle.dump(embed, open(emb_path + level, 'wb'))

    print(time.ctime(), 'Embedding of {} granularity attributes completed!'.format(emb_name))
    return embed


def embed_wd():
    attrs = pickle.load(open('../data/wd/attrs', 'rb'))
    g1, g2 = pickle.load(open('../data/wd/networks', 'rb'))
    print(time.ctime(), 'Size of the social user relationship graph:', len(g1), len(g2))
    char_attrs, word_attrs, article_attrs = [], [], []
    for i in range(len(attrs)):
        v = attrs[i]
        char_attrs.append(v[0])
        word_attrs.append(v[1])
        article_attrs.append(v[2])

    label_attrs = char_cn(get_user_tag(article_attrs))
    char_attrs = char_cn(char_attrs)
    word_attrs = word_cn(word_attrs)
    article_attrs_pre = article_cn_pre(article_attrs)
    article_attrs_lda = article_cn_lda(article_attrs)

    path_class = path_conf.wd
    for seed in range(1):
        for dim in [100]:
            emb_path = '../emb/emb_wd_seed_{}_dim_{}/'.format(seed, dim)
            if not os.path.exists(emb_path): os.mkdir(emb_path)

            # Embedding section, do not embed every time to save time, embed only the necessary parts
            article_embed_PRE(article_attrs_pre, dim, emb_path, path_class)
            article_embed_LDA(article_attrs_lda, dim, emb_path)
            char_embd(label_attrs, dim, emb_path, 'emb_l')
            char_embd(char_attrs, dim, emb_path, 'emb_c')
            word_embd(word_attrs, dim, g1, g2, emb_path, path_class)
            graph_embed(g1, dim, 'g1', emb_path)
            graph_embed(g2, dim, 'g2', emb_path)
            # Embedding section completed

            emb_g1 = pickle.load(open(emb_path + 'emb_g1', 'rb'))
            emb_g2 = pickle.load(open(emb_path + 'emb_g2', 'rb'))
            emb_g1.update(emb_g2)
            emb_c = pickle.load(open(emb_path + 'emb_c', 'rb'))
            emb_l = pickle.load(open(emb_path + 'emb_l', 'rb'))
            emb_w = pickle.load(open(emb_path + 'emb_w', 'rb'))
            emb_a_pre = pickle.load(open(emb_path + 'emb_a_pre', 'rb'))
            emb_a_lda = pickle.load(open(emb_path + 'emb_a_lda', 'rb'))
            emb_g = np.array([emb_g1[str(i)] for i in range(len(emb_g1))])

            # Standardize
            emb_c = (emb_c - np.mean(emb_c, axis=0, keepdims=True)) / np.std(emb_c, axis=0, keepdims=True)
            emb_l = (emb_l - np.mean(emb_l, axis=0, keepdims=True)) / np.std(emb_l, axis=0, keepdims=True)
            emb_w = (emb_w - np.mean(emb_w, axis=0, keepdims=True)) / np.std(emb_w, axis=0, keepdims=True)
            emb_a_pre = (emb_a_pre - np.mean(emb_a_pre, axis=0, keepdims=True)) / np.std(emb_a_pre, axis=0,
                                                                                         keepdims=True)
            emb_a_lda = (emb_a_lda - np.mean(emb_a_lda, axis=0, keepdims=True)) / np.std(emb_a_lda, axis=0,
                                                                                         keepdims=True)
            emb_g = (emb_g - np.mean(emb_g, axis=0, keepdims=True)) / np.std(emb_g, axis=0, keepdims=True)

            # Save embedding information
            pickle.dump((emb_c, emb_l, emb_w, emb_a_pre, emb_a_lda, emb_g),
                        open(emb_path + 'emb', 'wb'))


def embed_dblp():
    attrs = pickle.load(open('../data/dblp/attrs', 'rb'))
    g1, g2 = pickle.load(open('../data/dblp/networks', 'rb'))
    print(time.ctime(), 'Size of the social user relationship graph:', len(g1), len(g2))
    char_attrs, word_attrs, article_attrs = [], [], []
    for i in range(len(attrs)):
        v = attrs[i]
        char_attrs.append(v[0])
        word_attrs.append(v[1])
        article_attrs.append(v[2])

    article_attrs_pre = article_attrs  # No need for tokenization and stop word removal when using pre-trained model
    path_class = path_conf.dblp
    for seed in range(1):
        for dim in [100]:
            emb_path = '../emb/emb_dblp_seed_{}_dim_{}/'.format(seed, dim)

            if not os.path.exists(emb_path): os.mkdir(emb_path)

            # Embedding section, do not embed every time to save time, embed only the necessary parts
            article_embed_PRE(article_attrs_pre, dim, emb_path, path_class)
            # article_embed_LDA(article_attrs_lda, dim, emb_path)
            # char_embd(label_attrs, dim, emb_path, 'emb_l')
            # char_embd(char_attrs, dim, emb_path, 'emb_c')
            # word_embd(word_attrs, dim, g1, g2, emb_path, path_class)
            # graph_embed(g1, dim, 'g1', emb_path)
            # graph_embed(g2, dim, 'g2', emb_path)
            # Embedding section completed

            emb_g1 = pickle.load(open(emb_path + 'emb_g1', 'rb'))
            emb_g2 = pickle.load(open(emb_path + 'emb_g2', 'rb'))
            emb_g1.update(emb_g2)

            emb_c = pickle.load(open(emb_path + 'emb_c', 'rb'))
            emb_l = pickle.load(open(emb_path + 'emb_l', 'rb'))
            emb_w = pickle.load(open(emb_path + 'emb_w', 'rb'))
            emb_a_pre = pickle.load(open(emb_path + 'emb_a_pre', 'rb'))
            emb_a_lda = pickle.load(open(emb_path + 'emb_a_lda', 'rb'))
            emb_g = np.array([emb_g1[str(i)] for i in range(len(emb_g1))])

            # Standardize
            emb_c = (emb_c - np.mean(emb_c, axis=0, keepdims=True)) / np.std(emb_c, axis=0, keepdims=True)
            emb_l = (emb_l - np.mean(emb_l, axis=0, keepdims=True)) / np.std(emb_l, axis=0, keepdims=True)
            emb_w = (emb_w - np.mean(emb_w, axis=0, keepdims=True)) / np.std(emb_w, axis=0, keepdims=True)
            emb_a_pre = (emb_a_pre - np.mean(emb_a_pre, axis=0, keepdims=True)) / np.std(emb_a_pre, axis=0,
                                                                                         keepdims=True)
            emb_a_lda = (emb_a_lda - np.mean(emb_a_lda, axis=0, keepdims=True)) / np.std(emb_a_lda, axis=0,
                                                                                         keepdims=True)
            emb_g = (emb_g - np.mean(emb_g, axis=0, keepdims=True)) / np.std(emb_g, axis=0, keepdims=True)

            # Save embedding information
            pickle.dump((emb_c, emb_l, emb_w, emb_a_pre, emb_a_lda, emb_g),
                        open(emb_path + 'emb', 'wb'))


if __name__ == '__main__':
    # embed_wd()
    embed_dblp()
