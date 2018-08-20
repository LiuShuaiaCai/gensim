## 用法示例
初始化和训练模型
```python
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pprint import pprint
from multiprocessing import cpu_count

documents = [TaggedDocument(doc, [i]) for i,doc in enumerate(common_texts)]
pprint(documents)
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=cpu_count())
model.save('doc2vec.model')
model = Doc2Vec.load('doc2vec.model')

# documnets格式
[TaggedDocument(words=['human', 'interface', 'computer'], tags=[0]),
 TaggedDocument(words=['survey', 'user', 'computer', 'system', 'response', 'time'], tags=[1]),
 TaggedDocument(words=['eps', 'user', 'interface', 'system'], tags=[2]),
 TaggedDocument(words=['system', 'human', 'system', 'eps'], tags=[3]),
 TaggedDocument(words=['user', 'response', 'time'], tags=[4]),
 TaggedDocument(words=['trees'], tags=[5]),
 TaggedDocument(words=['graph', 'trees'], tags=[6]),
 TaggedDocument(words=['graph', 'minors', 'trees'], tags=[7]),
 TaggedDocument(words=['graph', 'minors', 'survey'], tags=[8])]
```

如果您已完成模型训练（=不再更新，仅查询，减少内存使用），您可以执行以下操作：
```python
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
```

推断新文档的向量：
```python
vector = model.infer_vector(["system", "response"])
print(vector)

[ 0.04975459  0.00867725  0.05494388  0.08542755 -0.06840173]

sims = model.docvecs.most_similar([vector], topn=10)
for count, sim in sims:
    print(count, sim)

1 0.7185414433479309
3 0.6786500215530396
4 0.3268129229545593
8 0.2588410973548889
2 0.1486310511827469
5 0.13447263836860657
6 0.08006234467029572
0 -0.014483608305454254
7 -0.7235662937164307
```