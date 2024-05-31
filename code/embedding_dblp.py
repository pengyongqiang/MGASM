import os
import pickle
import time

import numpy as np

from data_utils import z_score_normalization
from entities import User
from embedding_mathod import char_embedding, word_embedding, lda_embedding, graph_embed, electra_embed_en


def embed_dblp():
    attrs = pickle.load(open('../dataset/dblp/attrs', 'rb'))
    g1, g2 = pickle.load(open('../dataset/dblp/networks', 'rb'))
    print(time.ctime(), 'Size of the social user relationship struc:', len(g1), len(g2))
    user_list = []
    for i in range(len(attrs)):
        user_list.append(User(attrs[i], 'dblp'))

    dim_list = [50, 100,150, 200, 250, 300]
    # dim_list = [100]
    for dim in dim_list:
        emb_path = f'../emb/dblp/emb_dim_{dim}/'
        if not os.path.exists(emb_path): os.makedirs(emb_path)
        print(f"正在进行嵌入，维度：{dim}")

        # print("正在嵌入字符粒度属性...")
        # char_emb = char_embedding(user_list, 'char', dim, 'en')
        # np.save(emb_path + "char.npy", char_emb)
        # print("字符粒度属性嵌入完成。")

        print("正在嵌入标签粒度属性...")
        label_emb = char_embedding(user_list, 'label', dim, 'en')
        np.save(emb_path + "label.npy", label_emb)
        print("标签粒度属性嵌入完成。")

        print("正在嵌入下词粒度属性...")
        word_emb = word_embedding(user_list, 'word', dim, g1, g2, 'en')
        np.save(emb_path + "word.npy", word_emb)
        print("词粒度属性嵌入完成。")

        # print("正在嵌入文章粒度属性...")
        # article_plm = electra_embed_en(user_list, 256, dim).numpy()
        # article_lda = lda_embedding(user_list, dim, 'en')
        # np.save(emb_path + "article_plm.npy", article_plm)
        # np.save(emb_path + "article_lda.npy", article_lda)
        # print("文章粒度属性嵌入完成。")
        #
        # print("正在嵌入网络结构...")
        # g1_emb = graph_embed(g1, dim)
        # g2_emb = graph_embed(g2, dim)
        # g1_emb.update(g2_emb)
        # graph = np.array([g1_emb[str(i)] for i in range(len(g1_emb))])
        # struc = z_score_normalization(graph)
        # np.save(emb_path + "struc.npy", struc)
        # print("网络结构嵌入完成。")


if __name__ == '__main__':
    embed_dblp()
