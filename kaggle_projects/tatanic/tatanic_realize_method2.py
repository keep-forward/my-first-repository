# coding:utf-8
import pandas as pd
import numpy as np

# 读取csv文件数据
train = pd.read_csv('D:/kaggle_projects/titanic/dataset/train.csv')
test = pd.read_csv('D:/kaggle_projects/titanic/dataset/test.csv')
# print(train.describe()) # 查看train的统计数据
# print(test.head(2)) # 查看前5行
# print(train.tail(3)) # 查看后5行

# Data Cleaning (数据清洗)
# 统计各个变量的缺失值情况
# print(train.isnull().sum()) # 训练数据集的缺失情况
# print('\n')
# print(test.isnull().sum())  # 测试数据集的缺失情况

'''
对training data，Age(177)\Cabin(687)\Embarked(2)三个特征变量有缺失值;
对test data，Age(86)\Fare(1)\Cabin(327)三个特征变量有缺失值;
如果是连续变量，可采用预测模型；
如果是离散变量，可取中位数或众数等。
'''

# 1、年龄可以取均值填补，或者取Name中同为Mr、或Mrs、或miss的均值进行相应填补
# train['Age']