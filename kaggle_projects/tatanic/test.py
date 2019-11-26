import pandas as pd
import numpy as np

'''
获取MR.所有人的均值
'''
def getAverageValue(dataFrame):
    sum = 0
    for row in dataFrame.iterrows():
        # 取一行
        if row['Name'].contains('Mr\.'):
            sum += row['Age']
        return sum


# df1 = pd.DataFrame(np.random.randn(4,4), index=['r1','r2','r3','r4'], columns=list('ABCD'))
# print(df1)

train = pd.read_csv('D:/kaggle_projects/titanic/dataset/train.csv') # 训练数据集
# print(train.tail())
# print(train.head())
# print(train.describe())
bool = train['Name'].str.contains('Mr\.')
filter_Mr_data = train['Name'][bool]
# print(filter_Mr_data)
print('totalAge of Mr is ', getAverageValue(train))
# print(train.Name.str.contains('Mr\.'))
print('end')
