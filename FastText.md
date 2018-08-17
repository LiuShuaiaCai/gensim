#### fasttext用途
+ 词汇表征学习
+ 文本分类

#### 词汇表征学习|词向量模型学习
```
    import fasttext
    
    # Skipgram model
    model = fasttext.skipgram('data.txt', 'model')
    print(model.words) # list of words in dictionary
    
    # CBOW model
    model = fasttext.cbow('data.txt', 'model')
    print(model.words) # list of words in dictionary
```

举个例子
```
    model = fasttext.skipgram('train.txt', 'model', lr=0.1, dim=300)
```


#### 文本分类
```
    lassifier = fasttext.supervised('data.train.txt', 'classifier.model', label_prefix='__label__')
```
调用模型用的API
```
    input_file     training file path (required)
    output         output file path (required)
    lr             learning rate [0.05]
    lr_update_rate change the rate of updates for the learning rate [100]
    dim            size of word vectors [100]
    ws             size of the context window [5]
    epoch          number of epochs [5]
    min_count      minimal number of word occurences [5]
    neg            number of negatives sampled [5]
    word_ngrams    max length of word ngram [1]
    loss           loss function {ns, hs, softmax} [ns]
    bucket         number of buckets [2000000]
    minn           min length of char ngram [3]
    maxn           max length of char ngram [6]
    thread         number of threads [12]
    t              sampling threshold [0.0001]
    silent         disable the log output from the C++ extension [1]
    encoding       specify input_file encoding [utf-8]
```
- minCount 5：单词出现少于5就丢弃  
- minn 最小长度的字符  
- maxn 最长长度的字符 
- t 采样阈值
- lr 学习率–epoch 迭代次数
- neg 负采样
- loss  loss function {ns,hs, softmax}
- dim 词向量维度 
- ws 窗口大小



分类器的属性和方法
```
    classifier.labels                  # List of labels
    classifier.label_prefix            # Prefix of the label
    classifier.dim                     # Size of word vector
    classifier.ws                      # Size of context window
    classifier.epoch                   # Number of epochs
    classifier.min_count               # Minimal number of word occurences
    classifier.neg                     # Number of negative sampled
    classifier.word_ngrams             # Max length of word ngram
    classifier.loss_name               # Loss function name
    classifier.bucket                  # Number of buckets
    classifier.minn                    # Min length of char ngram
    classifier.maxn                    # Max length of char ngram
    classifier.lr_update_rate          # Rate of updates for the learning rate
    classifier.t                       # Value of sampling threshold
    classifier.encoding                # Encoding that used by classifier
    classifier.test(filename, k)       # Test the classifier
    classifier.predict(texts, k)       # Predict the most likely label
    classifier.predict_proba(texts, k) # Predict the most likely label include their probability
```

#### 调试分析
```
    classifier = fasttext.supervised("lab3fenci.csv","lab3fenci.model",
    label_prefix="__label__",lr=0.1,epoch=100,dim=200,bucket=5000000)
    result = classifier.test("lab3fenci.csv")
    print(result.precision)
    print(result.recall)
    print(result.precisionprint)
```


#### 预测
```
    texts = ['example very long text 1', 'example very longtext 2']
    labels = classifier.predict(texts)
    print labels
    
    # Or with the probability
    labels = classifier.predict_proba(texts)
    print labels
```
