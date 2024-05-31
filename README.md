# MGASM
The code and dataset for paper "Multi-granularity attribute similarity model for user alignment across social platforms under pre-aligned data sparsity"

## Datasets
1. Weibo-Douban (WD)

First, a small number of users on the Douban platform have posted their Weibo accounts on their homepages. These users on distinguish two platforms are real user identity links, which can be used as pre-aligned user pairs in the UIL task. Second, the original data is prepared by crawling users' information pages, including users' attributes and their follower/followee relations. A clear benefit of data crawling in the Weibo platform could not be directly identified in this step. Weibo allows only a small part (two hundred) of follower/followee relations to be returned by crawlers. Hence, the relations that come from traditional Weibo crawling methods are quite incomplete. On the other hand, the size of Weibo network is enormous. The empirical treatment is to extract a subnet with common characteristics from the original Weibo network. We repeatedly remove the nodes with the degrees less than 2 or more than 1000. Then, the community discovery algorithm is performed to find the subnets with the typical social network characteristics, including the approximate power-law degree distribution and the high aggregation coefficient. Similar operations are carried out on the Douban network.

2. DBLP17-19 (DBLP)

Each author in DBLP has a unique key, which can be used as the ground truth for the UIL problem. In this study, the DBLP network backups of different periods, i.e.,2017-12-1 and 2018-12-1, were used as the aligned subjects in the UIL experiments. We select the Turing Award winner Yoshua Bengio as the center node in each network, and then delete any nodes more than three steps away from the center. Besides, the size of two DBLPs is reduced by discovering network communities and repeatedly deleting leaf nodes. The final DBLPs also enjoy the characteristics of the power-law distribution and the high aggregation coefficient.

## Environment
We have saved all dependency information in `requirements.txt`. You can install these dependencies by running `pip install -r requirements.txt`.

## Usage
1. We have saved all embedding results in the `emb` folder. Therefore, you can directly run the `main.py` file to obtain the experiment results. You can adjust the `main` function in `main.py` according to your needs to control the content of the experiment.
2. If you want to re-execute the embedding process, you can run `embedding_dblp.py` and `embedding_wd.py` to complete the embeddings for the DBLP and WD datasets, respectively.

For the WD dataset, a Chinese Wikipedia corpus is needed to train the word vectors. Due to GitHub's file size restrictions, you can download the compressed file named "zhwiki-latest-pages-articles.xml.bz2" from the official Wikipedia website as the text corpus. Please place this file in the MGASM/models/cn directory. Of course, if you are not performing embedding, you can ignore this step. If you have any questions, please send an email to the address provided in the paper.

[//]: # (## Citation)

[//]: # (If you found this model or code useful, please cite it as follows:      )

[//]: # (```)

[//]: # (wating...)

[//]: # (```)

wating...)

[//]: # (```)

