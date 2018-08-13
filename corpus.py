# -*- ecoding: utf-8 -*-
from gensim import corpora
import jieba
from six import iteritems
from pprint import pprint

# 停用词
stoplists = set('的 与 和'.split())
# 所有词语的统计信息
lines = open('corpus.txt','r', encoding='utf-8')
dictionary = corpora.Dictionary(jieba.lcut(line) for line in lines)
# 除去停止词和仅出现一次的话
stop_ids = [ dictionary.token2id[stopword] for stopword in stoplists if stopword in dictionary.token2id ]
once_ids = [ tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1 ]
dictionary.filter_tokens(stop_ids + once_ids)
pprint(dictionary.token2id)
# 删除被删除的单词后的id序列中的空格,但是空格并没有去掉
dictionary.compactify()
pprint(dictionary.token2id)

for tokenid, docfreq in dictionary.dfs.items():
    print(tokenid, docfreq)


import gensim
import numpy as np
np_matrix = np.random.randint(10, size=[5,2])
print(np_matrix)
corpus = gensim.matutils.Dense2Corpus(np_matrix)
print(list(corpus))
np_matrix = gensim.matutils.corpus2dense(corpus, num_terms = 'number_of_corpus_features')
print(np_matrix)


import scipy
ss_matrix = scipy.sparse.random(5,2)
pprint(ss_matrix)
corpus = gensim.matutils.Sparse2Corpus(ss_matrix)
pprint(list(corpus))
csc_matrix = gensim.matutils.corpus2csc(corpus)
pprint(list(csc_matrix))