import pandas as pd
import matplotlib as plt
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np

# 读取csv文件 (以下两种写法均可)
# X_train = pd.read_csv('D:\\kaggle_projects\\titanic\\dataset\\train.csv')
train = pd.read_csv('D:/kaggle_projects/titanic/dataset/train.csv') # 训练数据集
test = pd.read_csv('D:/kaggle_projects/titanic/dataset/test.csv') # 测试数据集

# 查看数据集中各特征的缺失值数量
# train_feature_isnull_num = train.isnull().sum()
# print(train_feature_isnull_num)

'''
提取特征：
由于PassengerId是冗余信息，Name、Ticket和Cabin对乘客是否存活无明显影响，所以这四个不作为特征处理
'''
X_train = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
X_test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

y_train = train['Survived']

'''
缺失数据填充：
age:可以取中位数插补； 
cabin：有数据的存活率远比无数据的存活率高，可以将有无cabin数据作为特征；
embarked:可以取众数（出现频次高的值）插补；

#均值
np.mean(nums)
#中位数
np.median(nums)
求取众数
counts = np.bincount(nums)
#返回众数
np.argmax(counts)
'''
X_train['Embarked'].fillna('S') # 填充Embarked列缺失值，取众数
X_train['Age'].fillna(X_train['Age'].mean()) # 填充Age列缺失值，取平均数

'''
测试数据集缺失数据填充：
Age : 用均值填补
Fare ： 用均值填补
Cabin： 不做处理
'''

X_test['Age'].fillna(X_test['Age'].mean())
X_test['Fare'].fillna(X_test['Fare'].mean())

'''
探索可视化：(略)
女人和男人的存活率可视化
'''

'''
用DictVectorizer进行分类变量特征提取，将dict类型的list数据，装换成numpy array
'''
dict_vec = DictVectorizer(sparse=False)
X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
X_test = dict_vec.transform(X_test.to_dict(orient='record'))

'''
训练模型选择：
选择XGBOOST，拟合效果好
'''
xgb_model = xgb.XGBClassifier()

# 设置参数
params = dict(booster='gbtree',
              objective='multi:softmax',
              num_class=2,
              learning_rate=0.1,
              max_depth=2,
              silent=0,)
# 迭代次数
plst = list(params.items())
num_rounds = 1000

'''
#使用sklearn.model_selection中的train_test_split进行训练数据集的划分;
train_test_split使用介绍：
格式：x_train,x_valid,y_train,y_valid=train_test_split(x, y, test_size=0.2, random_state=1)
其中
x为待划分的样本特征集合；
y为待划分的样本标签；
test_size:若在0-1之间，表示测试验证集样本数目与原始数据集数目的比；若为整数，则是测试验证集的样本数目；
random_state：为随机数种子，其实就是该组随机数的编号；每次填1，其他参数一样的情况下得到的随机数组也是一样的；
    但填0或者不填，每次都不一样。

x_train：为划分出的训练集数据；
x_valid：为划分出的测试验证集的数据；
y_train：为划分出的训练集标签；
y_valid：为划分出的测试验证集标签；

'''
train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
print('train test split end')
# 矩阵赋值
xgb_valid = xgb.DMatrix(val_x, label=val_y)
print('print xgb_valid end ')
xgb_train = xgb.DMatrix(train_x, label=train_y)
print('print xgb_train end ')
xgb_test = xgb.DMatrix(X_test)
print('print xgb_test end ')

'''
训练模型：
early_stopping_rounds 当设置的迭代次数较大时，
early_stopping_rounds可在设置的迭代次数内准确率没有提升就停止训练
'''
# watchlist 方便查看运行情况
watchlist = [(xgb_train, 'train'), (xgb_valid, 'valid')]
model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)
print('train model end')
# 测试集合预测值
preds = model.predict(xgb_test, ntree_limit=model.best_ntree_limit)

# 结果输出
np.savetxt('D:/kaggle_projects/titanic/result/xgboost_result.csv', np.c_[range(892, len(X_test)+892), preds], delimiter=',', comments='', fmt='%d')
print('print predict result end')
