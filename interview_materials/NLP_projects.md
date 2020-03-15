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

公式及直观解释。



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

