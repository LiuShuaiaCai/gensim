from gensim import models, corpora, similarities
import jieba

# 加载词典、词库
dictionary = corpora.Dictionary.load('corpus.dict')
corpus = corpora.MmCorpus('corpus.mm')
# 转化
tfidf_model = models.TfidfModel(corpus)
corpus_tfidf = tfidf_model[corpus]
lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=200)
corpus_lsi = lsi_model[corpus_tfidf]

# 初始化查询结构
index = similarities.MatrixSimilarity(corpus_lsi)
# print(list(index))

# 执行查询
doc = '长沙市公安局官方微博@乌克兰长沙警事发布消息称'
vec_bow = dictionary.doc2bow(jieba.lcut(doc))
vec_lsi = lsi_model[vec_bow]

sims = index[vec_lsi]
# print(sims)

sims = sorted(enumerate(sims), key = lambda item: -item[1])
print(sims)