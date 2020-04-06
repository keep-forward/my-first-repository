# iNLP项目总结

## 一、文本情感分析

### 1. 项目介绍

kaggle项目：Bag of Words Meets Bags of Popcorn（电影评论文本情感分析）

kaggle地址： https://www.kaggle.com/c/word2vec-nlp-tutorial/data 



### 2. 数据细节

面试点：样本平均长度、词表大小、词频分布、样本均衡性。

采用的数据集：采用IMDB（Internet Movie DataBase，互联网电影资料库）的电影评论来做**情感分析**。该数据集包含10万条电影评论。25000条已标记的训练数据（评论的情感是二分类的，评分小于5的标记为0；评分大于7的标记为1）；50000条未标记的训练数据；25000条测试数据。

labeledTrainData.tsv，已标记的25000条训练数据，包含id、sentiment、review。

unlabeledTrainData.tsv，50000条未标记的训练数据，包含id、review。

testData.tsv，25000条测试数据，包含id、review。

处理流程主要包含以下几个模块:

利用pd.read_csv读取数据 --> 利用BeautifulSoup包去除评论中的HTML标签 --> 用正则化re去除评论中的标点符号 --> 将评论中所有大写字母换成小写，然后分割成独立的单词 -->  去除停用词。



### 3. 经典算法

公式及直观解释。

- BOW
- word2vec



### 4. 工程细节

特征构造思路、词向量是否预训练，语料不够怎么办。



### 5. 算法原理





### 6. 如何评估结果

给出precision/recall的定义及物理解释。



### 7. 除了自己用到的方法，还了解哪些方法；他们各自有哪些优缺点。





## 二、文本分类

### 1. 项目介绍

kaggle项目：Bag of Words Meets Bags of Popcorn（电影评论文本情感分析）

kaggle地址： https://www.kaggle.com/c/word2vec-nlp-tutorial/data 



### 2. 数据细节

面试点：样本平均长度、词表大小、词频分布、样本均衡性。

采用的数据集：采用IMDB（Internet Movie DataBase，互联网电影资料库）的电影评论来做**情感分析**。该数据集包含10万条电影评论。25000条已标记的训练数据（评论的情感是二分类的，评分小于5的标记为0；评分大于7的标记为1）；50000条未标记的训练数据；25000条测试数据。

labeledTrainData.tsv，已标记的25000条训练数据，包含id、sentiment、review。

unlabeledTrainData.tsv，50000条未标记的训练数据，包含id、review。

testData.tsv，25000条测试数据，包含id、review。

处理流程主要包含以下几个模块:

利用pd.read_csv读取数据 --> 利用BeautifulSoup包去除评论中的HTML标签 --> 用正则化re去除评论中的标点符号 --> 将评论中所有大写字母换成小写，然后分割成独立的单词 -->  去除停用词。



### 3. 经典算法

1. **fastText:**

   - 简介：是一个快速文本分类算法，与基于神经网络的分类算法相比有两大优点：

     1. fastText在保持高精度的情况下加快了训练速度和测试速度；
     2. 不需要预训练好的词向量，fastText会自己训练词向量；
     3. 两个重要优化：Hierarchical softmax、N-gram

   - 模型架构：

     与word2vecc中的CBOW很相似，不同之处是fastText预测标签，而CBOW预测的是中间词，即模型架构类似但模型的任务不同。

   - 为什么fastText甚至可以为语料库中未出现的单词产生词向量？

     因为fastText是通过包含在单词中的子字符sbustring of character来构建单词的词向量，正文中也有论述，因此这种训练模型的方式使得fastText可以为拼写错误的单词或者连接组装的单词产生词向量。

   - 可以在GPU上运行fastText项目吗？

     目前fastText仅仅可运行在CPU上，但这也是其优势所在，fastText的目的便是要成为一个高效的CPU上的分类模型。

   - 可以使用python语言或者其他语言使用fastText吗？

     目前在Github上有很少的关于fastText的其他语言实现的非官方版本，但是可以用TensorFlow实现的。

   - 可以在连续的数据集上使用fastText吗？

     不可以，fastText仅仅是用于离散的数据集，因此无法直接在连续的数据集上使用，但是可以将连续的数据离散化后使用fastText。

   - 数据中存在拼写错误，我们需要对文本进行规范化处理吗？

     如果出现的频率不高，没有必要，对模型效果不会有什么影响。

   - 在模型训练时遇到了NaN，为什么会这样？

     这种现象是可能出现的，很大原因是因为你的学习率太高了，可以尝试降低一下学习率直到不再出现NaN。

   - 

2. 



参考： https://www.jianshu.com/p/f69e8a306862 



### 4. 工程细节

特征构造思路、词向量是否预训练，语料不够怎么办。



### 5. 算法原理

mini-batch方式训练如何处理变长序列，重要参数对效果的影响



### 6. 如何评估结果

给出precision/recall的定义及物理解释。



### 7. 除了自己用到的方法，还了解哪些方法；他们各自有哪些优缺点。



## 三、命名实体识别（主要是BiLSTM+CRF的应用）

### 1. 项目介绍

kaggle项目：Bag of Words Meets Bags of Popcorn（电影评论文本情感分析）

kaggle地址： https://www.kaggle.com/c/word2vec-nlp-tutorial/data 



### 2. 数据细节

面试点：样本平均长度、词表大小、词频分布、样本均衡性。

采用的数据集：采用IMDB（Internet Movie DataBase，互联网电影资料库）的电影评论来做**情感分析**。该数据集包含10万条电影评论。25000条已标记的训练数据（评论的情感是二分类的，评分小于5的标记为0；评分大于7的标记为1）；50000条未标记的训练数据；25000条测试数据。

labeledTrainData.tsv，已标记的25000条训练数据，包含id、sentiment、review。

unlabeledTrainData.tsv，50000条未标记的训练数据，包含id、review。

testData.tsv，25000条测试数据，包含id、review。

处理流程主要包含以下几个模块:

利用pd.read_csv读取数据 --> 利用BeautifulSoup包去除评论中的HTML标签 --> 用正则化re去除评论中的标点符号 --> 将评论中所有大写字母换成小写，然后分割成独立的单词 -->  去除停用词。



### 3. 经典算法

公式及直观解释。



### 4. 工程细节

特征构造思路、词向量是否预训练，语料不够怎么办。



### 5. 算法原理

mini-batch方式训练如何处理变长序列，重要参数对效果的影响



### 6. 如何评估结果

给出precision/recall的定义及物理解释。



### 7. 除了自己用到的方法，还了解哪些方法；他们各自有哪些优缺点。

