class cn:
    # 维基原始语料位置
    input_corpus_file = '../models/cn/zhwiki-latest-pages-articles.xml.bz2'
    # 构建的维基语料库位置
    out_corpus_file = '../models/cn/zhwiki_corpus'
    # word2vec模型嵌入保存地址
    wv_save_file = '../models/cn/wv/'
    # 用于嵌入中文文章的预训练模型
    chinese_electra = '../models/pmodels/chinese_electra'


class en:
    # word2vec模型嵌入保存地址
    wv_save_file = '../models/en/wv/'
    # 用于嵌入英文文章的预训练模型
    english_electra = '../models/pmodels/electra'
