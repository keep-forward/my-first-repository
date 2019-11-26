import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt   # 引入时需注意，是pyplot而不是matplotlib
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('D:\kaggle_projects\house_prices\dataset\\train.csv')
# 查看dataframe的基本统计量，查看目标数据（房价）的宏观范围
# print(train['SalePrice'].describe())
# print(train['GrLivArea'].describe())

# 绘制变量和目标值的关系，确定哪些是离群点，随后移除离群点
# 绘制图-方法一
# var = 'GrLivArea'
# data = pd.concat([train['SalePrice'], train[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

# 绘制图-方法二
# plt.scatter(train.GrLivArea, train.SalePrice, c="blue", marker="s")
# plt.title("Looking for outliers")
# plt.xlabel("GrLivArea")
# plt.ylabel("SalePrice")
# plt.show()

# 可以确定GrLivArea大于4000，以及
# 热力图：观察哪些变量会和预测目标关系比较大，哪些变量之间会有较强的关联
# corrmat = train.corr()
# f, ax = plt.subplots(figsize=(12,9))
# sns.heatmap(corrmat,vmax=.8, square=True)
# plt.show()
'''
查看变量之间的关系
'''
# sns.set()
# cols = ['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']
# sns.pairplot(train[cols], size=2.5)
# plt.show()
'''
处理missing data:
（1）缺失值的percent（缺失个数/该特征变量总数（即样本数））超过15%的可以删除该特征变量；
（2）其他，按具体情况如对结果影响不大的变量删除、用均值或者中位数填补等方法进行处理
'''
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
#print('train.isnull().sum() is \n', train.isnull().sum())
#print('train.isnull().count() is \n', train.isnull().count())
missing_data = pd.concat([total,percent], axis=1, keys=['Total','Percent'])
#print(missing_data.head(20))
'''
dataFrame.index：获取dataFrame的行变量
'''
# print((missing_data[missing_data['Total']>1]).index)
# print('before drop \n', train)
# print(train.drop((missing_data[missing_data['Total']>1]).index, 1))
# print('afer drop \n', len(train.columns))
train = train.drop((missing_data[missing_data['Total']>1]).index, 1)
train = train.drop((train.loc[train['Electrical'].isnull()]).index)
train.isnull().sum().max()
'''
fit_transform():标准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而影响；
                先拟合数据，再将其标准化
transform()：通过找中心和缩放等实现数据标准化

为了数据归一化(是特征数据方差为1，均值为0)，我们需要计算出特征数据的均值和方差，再使用公式进行归一化。
'''
ss = StandardScaler()  # 必须加括号()
saleprice_scaled = ss.fit_transform(train['SalePrice'][:,np.newaxis])
'''
argsort()：将序列按照从小到大排序，取其索引输出
'''
sorted_data = saleprice_scaled[saleprice_scaled[:,0].argsort()]
print('outer range (low) of the distribution \n')
print(sorted_data[:10])
print('\n outer range (high) of the distribution \n')
print(sorted_data[-10:])
print('end')

'''
Bivarivate analysis：
concat函数是pandas的方法，可以将数据根据不同的轴做简单的融合
'''
var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
# ylim表示y轴的取值范围；xlim表示x轴的取值范围
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
plt.show()