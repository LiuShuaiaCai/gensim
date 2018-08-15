# -*- encoding: utf-8 -*-
from gensim import models, corpora, similarities
from pprint import pprint
import jieba
from collections import defaultdict
import os

# 获取字典和语料库
def GetDictionaryCorpus():
    if os.path.exists('corpus.dict') and os.path.exists('corpus.mm'):
        dictionary = corpora.Dictionary.load('corpus.dict')
        corpus = corpora.MmCorpus('corpus.mm')
    else:
        # 语料样本
        documents = [
            '0南京江心洲污泥偷排”等污泥偷排或处置不当而造成的污染问题，不断被媒体曝光',
            '1面对美国金融危机冲击与国内经济增速下滑形势，中国政府在2008年11月初快速推出“4万亿”投资十项措施',
            '2全国大面积出现的雾霾，使解决我国环境质量恶化问题的紧迫性得到全社会的广泛关注',
            '3大约是1962年的夏天吧，潘文突然出现在我们居住的安宁巷中，她旁边走着40号王孃孃家的大儿子，一看就知道，他们是一对恋人。那时候，潘文梳着一条长长的独辫',
            '4坐落在美国科罗拉多州的小镇蒙特苏马有一座4200平方英尺(约合390平方米)的房子，该建筑外表上与普通民居毫无区别，但其内在构造却别有洞天',
            '5据英国《每日邮报》报道，美国威斯康辛州的非营利组织“占领麦迪逊建筑公司”(OMBuild)在华盛顿和俄勒冈州打造了99平方英尺(约9平方米)的迷你房屋',
            '6长沙市公安局官方微博@长沙警事发布消息称，3月14日上午10时15分许，长沙市开福区伍家岭沙湖桥菜市场内，两名摊贩因纠纷引发互殴，其中一人被对方砍死',
            '7乌克兰克里米亚就留在乌克兰还是加入俄罗斯举行全民公投，全部选票的统计结果表明，96.6%的选民赞成克里米亚加入俄罗斯，但未获得乌克兰和国际社会的普遍承认',
            '8京津冀的大气污染，造成了巨大的综合负面效应，显性的是空气污染、水质变差、交通拥堵、食品不安全等，隐性的是各种恶性疾病的患者增加，生存环境越来越差',
            '9 1954年2月19日，苏联最高苏维埃主席团，在“兄弟的乌克兰与俄罗斯结盟300周年之际”通过决议，将俄罗斯联邦的克里米亚州，划归乌克兰加盟共和国',
            '10北京市昌平区一航空训练基地，演练人员身穿训练服，从机舱逃生门滑降到地面',
            '11腾讯入股京东的公告如期而至，与三周前的传闻吻合。毫无疑问，仅仅是传闻阶段的“联姻”，已经改变了京东赴美上市的舆论氛围',
            '12国防部网站消息，3月8日凌晨，马来西亚航空公司MH370航班起飞后与地面失去联系，西安卫星测控中心在第一时间启动应急机制，配合地面搜救人员开展对失联航班的搜索救援行动',
            '13新华社昆明3月2日电，记者从昆明市政府新闻办获悉，昆明“3·01”事件事发现场证据表明，这是一起由新疆分裂势力一手策划组织的严重暴力恐怖事件',
            '14在即将召开的全国“两会”上，中国政府将提出2014年GDP增长7.5%左右、CPI通胀率控制在3.5%的目标',
            '15中共中央总书记、国家主席、中央军委主席习近平看望出席全国政协十二届二次会议的委员并参加分组讨论时强调，团结稳定是福，分裂动乱是祸。全国各族人民都要珍惜民族大团结的政治局面，都要坚决反对一切危害各民族大团结的言行'
        ]
        # 停用词
        stoplists = ['，','。','《','》','@','“','”','(',')','、','的']
        # 将语料库进行分词
        texts = [ [ word for word in jieba.lcut(document) if word not in stoplists ] for document in documents ]
        # 统计词频
        frequency = defaultdict(int)
        for text in texts:
            for word in text:
                frequency[word] += 1
        # 删除词频为1的词（根据需要处理，这里只是演示一下处理的方法）
        new_texts = [ [ token for token in text if frequency[token] > 1 ] for text in texts ]
        
        # 生成词典
        dictionary = corpora.Dictionary(new_texts, prune_at=2000000)
        dictionary.save('corpus.dict')
        # 生成词库，以（词， 词频）的方式存储
        corpus = [ dictionary.doc2bow(text) for text in new_texts ]
        corpora.MmCorpus.serialize('corpus.mm', corpus)
    return dictionary, corpus

# TF_IDF
def tfidf():
    dictionary, corpus = GetDictionaryCorpus()
    # 将语料转化为tfidf,tfidf被视为一个只读对象，可以用于将任何向量从旧表示（词频）转换为新表示（TfIdf实值权重）
    tfidf_model = models.TfidfModel(corpus)
    # 使用模型tfidf_model，将doc_bow(由词,词频)表示转换成(词,tfidf)表示
    doc_bow = [(0,1),(1,1)]
    tfidf = tfidf_model[doc_bow]

    # 转化整个词库
    corpus_tfidf = tfidf_model[corpus]
    # pprint(list(corpus_tfidf))
    return corpus_tfidf

# LDA
def lda():
    dictionary, corpus = GetDictionaryCorpus()
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=200)
    topics = lda_model.print_topics()
    print(topics)

# LSI
def lsi():
    '''
    # 潜在语义索引(Latent Semantic Indexing,以下简称LSI)，有的文章也叫Latent Semantic  Analysis（LSA）
    # LSI是基于奇异值分解（SVD）的方法来得到文本的主题的
    '''
    dictionary, corpus = GetDictionaryCorpus()
    corpus_tfidf = tfidf()
    # 转化为LSI(在这里实际执行了bow-> tfidf和tfidf-> lsi转换)
    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=200)
    corpus_lsi = lsi_model[corpus_tfidf]
    # pprint(list(corpus_lsi))
 

# RP
def rp():
    '''
    随机投影(Random Projections)，RP旨在减少矢量空间维数。
    这是非常有效的方法，通过投掷一点随机性来近似文档之间的TfIdf距离。
    推荐的目标维度数百/千，取决于您的数据集。 
    '''
    corpus_tfidf = tfidf()
    rp_model = models.RpModel(corpus_tfidf, num_topics=2)
    corpus_rp = rp_model[corpus_tfidf]
    pprint(list(corpus_rp))

rp()

