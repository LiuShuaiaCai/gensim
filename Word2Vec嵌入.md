## Word2Vec 嵌入

### 背景
word2vec是Google在2013年推出开源的NLP工具。它的特点就是将所有的词向量化，这样词与词之间就可以定量的去度量它们之间的关系，挖掘词之间的联系

### 算法
Word2Vec的算法包括 skip-gram 和 cbow 模型，使用分层SOFTMAX或负采样

### 嵌入方法
在Gensim中训练单词向量的方法除了Word2Vec。还有[Doc2Vec](./Doc2Vec.md)，[FastText](./FastText.md)包装VarEmbed和WordRank。

### 用法示例
使用用例，初始化模型：
```python
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from multiprocessing import cpu_count

model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=cpu_count())
model.save('word2vec.model')
```
流式传输培训，意味着句子可以是生成器，即时从磁盘读取输入数据，而无需将整个语料库加载到RAM中。

这也意味着您可以在以后继续训练模型：
```python
model = Word2Vec.load('word2vec.model')
model.train([["hello", "world"]], total_examples=1, epochs=1)
```
训练过的单词向量存储在model.wv中的KeyedVectors实例中：
```python
vector = model.wv['computer']
print(vector)
```

### 常用方法
class gensim.models.word2vec.LineSentence（source，max_sentence_length = 10000，limit = None ）¶   
迭代包含句子的文件：一行是一个句子。单词必须已经过预处理并由空格分隔。
作用就是把文本转化为列表，看下面的小例子：
```python
from gensim.test.utils import datapath
from gensim.models.word2vec import LineSentence
sentences = LineSentence(datapath('lee_background.cor'))
for sentence in sentences:
    print(sentence)
```

gensim.models.word2vec.PathLineSentences（source，max_sentence_length = 10000，limit = None ）¶  
和LineSentence类似，但按文件名的字母顺序处理目录中的所有文件。
该目录必须只包含可以通过以下方式读取的文件gensim.models.word2vec.LineSentence：.bz2，.gz和文本文件。任何不以.bz2或.gz结尾的文件都被假定为文本文件。

gensim.models.word2vec.Word2Vec(sentences=None, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=<built-in function hash>, iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000, compute_loss=False, callbacks=(), max_final_vocab=None)¶  
> Word2Vec的参数：
+ sentences：可迭代的句子可以简单地是令牌列表的列表，但是对于更大的语料库，考虑直接从磁盘/网络流式传输句子的迭代。 有关此类示例，请参阅word2vec模块中的BrownCorpus，Text8Corpus或LineSentence。 另请参阅Python中的数据流教程。 如果您不提供句子，则模型将保持未初始化状态 - 如果您计划以其他方式初始化该模型，请使用该模型。
+ size：单词向量的维度
+ window：句子中当前和预测单词之间的最大距离
+ min_count：忽略总频率低于此值的所有单词
+ workers：使用这些工作线程来训练模型（=使用多核机器进行更快的训练）
+ sg：训练的方法：1为skip-gram,2为cbow
+ hs：如果为1，分层softmax将用于模型训练。如果为0，negative不为0，则采用负采样
+ negative：如果大于0，将采用负采样，negative表示应绘制多少“噪声词”（通常在5-20之间）。如果设为0，则不使用负采样

🌰 例子
初始化并训练Word2Vec模型
```python
from gensim.models import Word2Vec
sentence = [["cat", "say", "meow"], ["dog", "say", "woof"]]
model = Word2Vec(sentence, min_count=1)
print(model)
```

### 模型的常用方法
> most_similar(positive = None，negative = None，topn = 10，restrict_vocab = None，indexer = None ）¶  

找到前N个最相似的单词。正面词对相似性有积极贡献，负面词有负面影响。

该方法计算给定单词的投影权重向量的简单平均值与模型中每个单词的向量之间的余弦相似度。该方法对应于原始word2vec实现中的单词类比和距离脚本。

参数：
+ positive：积极贡献的单词列表
+ negative：负面贡献的单词列表
+ topn：要返回的前N个相似单词的数量
+ restrct_vocab：它限制搜索最相似值的向量范围。例如，+ restrict_vocab = 10000只会检查词汇顺序中的前10000个单词向量。（如果您按降序频率对词汇表进行排序，这可能会有意义。）

> most_similar_cosmul(positive=None, negative=None, topn=10)¶  

和most_similar类似

> n_similarity（ws1，ws2 ）¶

计算两组单词之间的余弦相似度。

> similar_by_vector（vector，topn = 10，restrict_vocab = None ）¶

通过向量找到前N个最相似的单词。  
参数：  
+ vecor：要计算相似性的矢量
+ topn：要返回的前N个相似单词的数量。如果topn为False，则similar_by_vector返回相似性得分的向量。
+ restrict_vocab：可选的整数，它限制搜索最相似值的向量范围。例如，restrict_vocab = 10000只会检查词汇顺序中的前10000个单词向量。（如果您按降序频率对词汇表进行排序，这可能会有意义。）

> similar_by_word(word, topn=10, restrict_vocab = None )

通过单词找到前N个最相似的单词。  

> similarity(w1, w2)

计算两个单词之间的余弦相似度

> similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0, nonzero_limit=100, dtype=<type 'numpy.float32'>)

构造用于计算软余弦测量的术语相似度矩阵。

