class wd:
    # Wikipedia original corpus location
    input_corpus_file = '../data/wd/zhwiki-latest-pages-articles.xml.bz2'
    # Constructed Wikipedia corpus location
    out_corpus_file = '../data/wd/zhwiki_corpus'
    # Word2Vec model embedding save location
    wv_save_file = '../data/wd/models/word2vec.model'
    # LDA model embedding save location
    lda_save_file = '../data/wd/models/lda.model'
    # Graph embedding model save location
    eg_save_file = '../data/wd/models/'
    # Similarity model save location
    similar_model_save_file = '../data/wd/models/'


class dblp:
    # DBLP original corpus location
    input_corpus_file = '../data/dblp/zhwiki-latest-pages-articles.xml.bz2'
    # Constructed DBLP corpus location
    out_corpus_file = '../data/dblp/zhwiki_corpus'
    # Word2Vec model embedding save location
    wv_save_file = '../data/dblp/models/word2vec.model'
    # LDA model embedding save location
    lda_save_file = '../data/dblp/models/lda.model'
    # Graph embedding model save location
    eg_save_file = '../data/dblp/models/'
    # Similarity model save location
    similar_model_save_file = '../data/dblp/models/'


# Pretrained model locations
bert_base_cn = '../data/pmodels/bert_base_chinese'
bert_base = '../data/pmodels/bert_base_uncased'
chinese_electra = '../data/pmodels/chinese_electra'
bert_base_ner = '../data/pmodels/bert_base_ner'
