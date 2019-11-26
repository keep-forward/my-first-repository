import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette() # seaborn 调色盘
'''
seafborn 预先定义了5中主题背景样式，以适合不同场景的需要
分别是：darkgrid, whitegrid, dark, white, 和 ticks, 默认是darkgrid
'''
sns.set_style('darkgrid')
import warnings

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn # ignore annoying warning(from sklearn and seaborn)

from scipy import stats
from scipy.stats import norm, skew

# 输出保留三位小数 ， limiting floats output to 3 decimal points
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

'''
subprocess, 创建新的进程取执行另外的程序，并与它进行通信；
1、call(): 父进程等待子进程完成
2、check_call(): 
3、check_output():返回子进程向标准输出的输出结果
'''
# from subprocess import check_output
# # check the files available in the directory
# print(check_output(["ls", "../input"]).decode("utf8")) # D:/kaggle_projects/house_prices/dataset/
'''
执行上述语句报错“FileNotFound”，原因：windows不符合POSIX（可移植操作系统接口），如没有ls二进制文件，
因此，子进程无法找到该文件ls，从而抛出这个错误。
'''

train = pd.read_csv('D:\kaggle_projects\house_prices\dataset\\train.csv')
test = pd.read_csv('D:\kaggle_projects\house_prices\dataset\\test.csv')
'''
head(5),显示前5行
tail(5),显示后5行
上述两个函数都不带参数时，显示全部数据
'''
# print(train.head())
# print(test.head(10))
# check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

# save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

# Now drop the 'Id' column since it`s unnecessary for the prediction process
'''
drop函数，删除表中某一行或者某一列，不改变原有DataFrame中数据，而是返回另一个DataFrame来存放删除后的
数据；
该函数默认删除行（即axis=0）,而axis=1表示列方向上;
inplace参数：True时，原数组被直接替换； 
             false时，原数组名对应的内存值不变，需要将结果赋给一个新的数组名。
'''
train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)

# check again the data size after dropping the "Id" variable
print("\n The train data size after dropping Id feature is : {} ".format(train.shape))
print("\n The test data size after dropping Id feature is : {} ".format(test.shape))

'''
Data processing:
'''

# firstly, Let`s explore these outliers
# fig, ax = plt.subplots()
# ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('GrLivArea', fontsize=13)
# plt.show()

'''
以下两种表达方式等价：
（1）
fig = plt.figure()
fig.add_subplot(111)
(2)
fig,ax=plt.subplots()
'''

# Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

# check the graphic again
# fig, ax = plt.subplots()
# ax.scatter(train['GrLivArea'], train['SalePrice'])
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('GrLivArea', fontsize=13)
# plt.show()

# target variable
'''
seaborn的distplot参数
fit：控制拟合的参数分布图形，能够直观地评估它与观察数据的对应关系(黑色线条为确定的分布)
'''
# sns.distplot(train['SalePrice'], fit=norm); # 拟合标准正态分布
# # get the fitted parameters used by the function
# (mu, sigma) = norm.fit(train['SalePrice'])
# print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# now plot the distribution
# plt.legend 用于显示图例
# plt.legend(['Normal dist . ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)],
#            loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')

# get also the plot
# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)
# plt.show()

'''
transform target variable to make linear model more normally distributed,
now log-transformation of the target variable
'''
# we use the numpy function log1p which applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])

# check the new distribution
# sns.distplot(train["SalePrice"], fit=norm);
# (mu, sigma) = norm.fit(train['SalePrice'])
# print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)],
#            loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')

# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)
# plt.show()

'''
Feature Engineering
'''
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train,test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))

# missing data
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
# print(all_data_na)
missing_data = pd.DataFrame({"Missing Data" : all_data_na})
missing_data.head(20)
# print(missing_data.head(20))

# f, ax = plt.subplots(figsize=(15,12))
'''
设置x坐标轴的信息显示：
rotation设置做标注各列字体的显示旋转度，如horizontal是水平的，‘45’是与轴成45°角.
'''
# plt.xticks(rotation='90')
# 绘制柱状图
# sns.barplot(x=all_data_na.index, y=all_data_na)
# plt.xlabel('Features', fontsize=15)
# plt.ylabel('Percent of missing values', fontsize=15)
# plt.show()

'''
Data correlation:
使用热力图来展示数据表里面多个特征两两之间的相似度（关联性）
为了考察两个特征间的关系，可以借助随机变量的协方差（是对两个随机变量联合分布线性相关程度的一种度量）

'''
# Correlation map to see how features are correlated with SalePrice
# corrmat = train.corr() # 相关系数矩阵，给出任意两个变量之间的相关系数
# plt.subplots(figsize=(12,9))
# sns.heatmap(corrmat, vmax=0.9, square=True)
# plt.show()

'''
imputing missing values
'''
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
# median 取中值
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

all_data["MasVnrType"] = all_data["MasVnrType"].fillna('None')
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
# mode() 取众数
all_data["MSZoning"] = all_data["MSZoning"].fillna(all_data["MSZoning"].mode()[0])

# this feature "Utilities" won`t help in predictive modeling
all_data = all_data.drop(['Utilities'], axis=1)

all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data["Electrical"] = all_data["Electrical"].fillna(all_data["Electrical"].mode()[0])
all_data["KitchenQual"] = all_data["KitchenQual"].fillna(all_data["KitchenQual"].mode()[0])

all_data["Exterior1st"] = all_data["Exterior1st"].fillna(all_data["Exterior1st"].mode()[0])
all_data["Exterior2nd"] = all_data["Exterior2nd"].fillna(all_data["Exterior2nd"].mode()[0])

all_data["SaleType"] = all_data["SaleType"].fillna(all_data["SaleType"].mode()[0])
all_data["MSSubClass"] = all_data["MSSubClass"].fillna("None")

'''
check: is there any remaining missing value?
'''
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({"Missing Ratio" : all_data_na})
missing_data.head()
print("remaining missing data is \n")
print(missing_data.head())

'''
More features engeneering:
transforming some numerical variables that are categorical

astype(float): numpy的转换数据类型
apply(int)或apply(max) : 既可以进行数据类型转换，也可以应用于函数

几种数据类型：
Categorical Type(分类数据)：如皮肤颜色：黄色、白色和黑色等；均值、加减无意义
Ordinal Type（可排序数据）：如编程能力有初级、中级、高级；
Interval Type（间隔数据）：如工资、年龄；均值是有意义的
'''
all_data["MSSubClass"] = all_data["MSSubClass"].apply(str)

# changing OverallCond into a categorical variable
all_data["OverallCond"] = all_data["OverallCond"].astype(str)

# Year and month are transformed into categorical featrues
all_data["YrSold"] = all_data["YrSold"].astype(str)
all_data["MoSold"] = all_data["MoSold"].astype(str)

'''
LabelEncoder():标准化标签，将标签值统一转换成range(标签值个数-1)范围内:
    目的就是将object类型转化为数值
举例：
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(["Japan", "China", "Japan", "Korea", ""China])
print("标签个数:%s" % le.classes_)
print("标签值标准化：%s" % le.transform(["Japan", "China", "Japan", "Korea", ""China]))
print("标准化标签值反转：%s" % le.inverse_transform([0,2,0,1,2]))
'''
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')

# process columns, apply LabelEncode to categorical features
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))  # 匹配
    all_data[c]=lbl.transform(list(all_data[c].values))  # 标签值标准化

# shape
print('shape all_data: {}'.format(all_data.shape))

'''
adding one more important feature
'''
# adding total sqfoorage feature
all_data["TotalSF"] = all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]

# skewed features
'''
由于字符串长度不是固定的，pandas没有用字节字符串的形式，而是用object ndarray
'''
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
'''
偏度(skewness)：是衡量随机变量概率分布的不对称性，是相对于平均值（期望值）不对称程度的度量。
偏度值可以为正、为负或者无定义；为负时，长尾在左侧，数据集中在右侧；为正时，长尾在右侧,数据集中在左侧。
偏度为0就说明是标准的高斯分布。

计算公式：随机变量X的三阶中心距即为偏度
拓展：随机变量的一阶原点距（E（X））为期望；
      二阶中心距（E（|X-E（X）|的平方））为方差；
      三阶中心距（E（|X-E（X）|的立方））为偏度(Skewness)；
      四阶中心距（E（|X-E（X）|的四次方））为峰度(Kurtosis)；
      
'''
# check the skew of all_numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({"Skew" :skewed_feats})
skewed_feats.head(10)
# print(skewed_feats.head(10))

'''
Box Cox Transformation of (highly) skewed features:
样本分布不是正态分布的时候可以采用数据变换
Box-Cox 变换可以保证数据进行成功的正态变换，可以更好地满足正态性、独立性，减小线性回归的最小二乘估计系数的误差

计算方法： log，对数转换时使用最多的
      还有平方根转换、倒数转换、平方根后取倒数等等
'''
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
# (1+x)的Box-Cox 变换
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in  skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)

'''
Getting dummy categorical features:
用于特征提取:
one-hot encoding
one-hot的基本思想：将离散型特征的每一种取值都看成一种状态，
若你的这一特征中有N个不相同的取值，那么我们就可以将该特征抽象成N种不同的状态，
one-hot编码保证了每一个取值只会使得一种状态处于“激活态”，
也就是说这N种状态中只有一个状态位值为1，其他状态位都是0。

'''
all_data = pd.get_dummies(all_data)
print(all_data.shape)

train = all_data[:ntrain]
test = all_data[ntrain:]

'''
Moodelling
'''
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
# import lightgbm as lgb # lightgbm必须安装在64-bit的python中

'''
# define a cross validation strategy, use cross_val_score function of sklearn
1) train_test_split:对数据集进行快速打乱，分为训练集和测试集

2）cross_val_score:对数据集进行指定次数的交叉验证，并为每次验证效果评测

3）cross_val_predict:与cross_val_score很像，不同于后者返回评测效果，
                前者返回的是estimator的分类结果（或者回归值）

4）KFold:将数据集分成k份，k折就是将数据集通过k次分割，使得所有数据既在训练集中出现过，
                又在测试集中出现过，当然，每次分割中不会有重叠。相当于无放回抽样。
'''
n_folds = 5 # 将数据集划分成几等份，其中4份用来训练，另一份用来测试，共迭代5次，得到评分数组（5个值），然后求均值
def rmsle_cv(model):
    # KFold：生成交叉验证数据集，shuffle在每次划分时是否进行洗牌；random_state随机种子。
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    # print("n_folds is ",n_folds, "kf is ", kf)
    '''
    参数：
    model：模型对象；
    train.values：训练数据
    y_train:预测的标签数据
    scoring：调用的方法，neg_mean_squared_error指损失函数选用均方误差，反应估计量与被估计量之间的差异程度
    cv: 指交叉验证生成器可迭代的次数
    
    返回的是交叉验证每次运行的评分数组。
    '''
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
    return(rmse)

'''
Base models
'''

# （1）LASSO Regression:
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
# （2）Elastic Net Regression
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
# （3）Kernel Ridge Regression
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
# （4）Gradient Boosting Regression
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5)
# （5）XGBoost
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state=7, nthread=-1)

score = rmsle_cv(model_xgb)
# mean()是求均值（用μ表示）函数，std()是求标准差（用sigma表示）函数，方差是标准差的平方
print("\nmodel_xgb score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

'''
stacking model(叠加模型):
BaseEstimator:基本估计器
RegressorMixin:回归器的混合类
TransformerMixin:转换器的混合类
关于Mixin是什么，简单理解就是带有实现方法的借口，可以将其看做是组合模式的一种实现
'''

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
        return self

    # now we do the predictions for cloned models and average them
    def predict(self, X):
        # np.column_stack: 行不变列扩展
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

'''
Averaged base models score
'''
averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))
score = rmsle_cv(averaged_models)
print("Averaged base models score : {:.4f}({:.4f})\n".format(score.mean(), score.std()))

'''
Stacking averaged Models Class:Adding a Meta-model
'''
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # we again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        '''
        enumerate函数：将一个可遍历的数据对象组合为一个索引序列，同时
                        列出数据和下标
        '''
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    '''
    Do the predictions of all base models on the test data and use the averaged predictions as 
    meta-features for the final prediction which is done by the meta-model
    '''
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)

# Stacking Averaged models Score
stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR),
                                                 meta_model=lasso)
score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score : {:.4f}({:.4f})".format(score.mean(), score.std()))

'''
Ensembling StackingRegression, XGBoost and LifhtGBM(暂时实现不了)
'''

# define a rmsle evaluation function
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

# final Training and Prediction
# StackingRegression
stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))

# XGBoost
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))

# LightGBM 由于需要64-bit的Python才能安装，故不能实现

'''
RMSE on the entire Train data when averaging
'''
print("RMSLE score on train data :")
print(rmsle(y_train, stacked_train_pred*0.7 + xgb_train_pred*0.3))

# Ensemble prediction
ensemble = stacked_pred*0.7 + xgb_pred*0.3

'''
Submission
'''
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('D:\kaggle_projects\house_prices\submission\submission.csv', index=False)