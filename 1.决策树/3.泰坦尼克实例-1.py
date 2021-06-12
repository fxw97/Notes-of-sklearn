import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

data = pd.read_csv(r'3.titanic_data/train.csv')

# 特征选择,删掉不需要的列,axis=1表示删除列是必须传入的参数
data.drop(['Cabin','Name','Ticket'],axis=1,inplace=True)

# 处理缺失值
data['Age'] = data['Age'].fillna(data['Age'].mean())
data = data.dropna(axis=0) # 删除有缺失值的行

# 将字符串信息数字化
labels = data['Embarked'].unique().tolist()
data['Embarked'] = data['Embarked'].apply(lambda x:labels.index(x))

data['Sex'] = (data['Sex'] == 'male').astype('int')
# print(data.info())

x = data.iloc[:,data.columns != 'Survived']
y = data.loc[:,'Survived']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(x,y,test_size=0.3)

# 纠正索引:将分好之后数据的索引改为正序排序
for i in [Xtrain, Xtest, Ytrain, Ytest]:
    i.index = range(i.shape[0])

# clf = DecisionTreeClassifier(random_state=25)
# clf = clf.fit(Xtrain,Ytrain)
# score = cross_val_score(clf,x,y,cv=10).mean() 交叉验证结果均值为0.75

# 调参，寻找模拟更准确的参数
tr = []
te = []
for i in range(10):
    clf = DecisionTreeClassifier(random_state=25,max_depth=i+1,criterion='entropy')
    clf = clf.fit(Xtrain,Ytrain)
    score_tr = clf.score(Xtrain,Ytrain)
    score_te = cross_val_score(clf,x,y,cv=10).mean()
    tr.append(score_tr)
    te.append(score_te)
print(max(te))

# 若模型在训练集模拟结果比测试集高很多，说明模型有过拟合
plt.plot(range(1,11),tr,c='r',label='train')
plt.plot(range(1,11),te,c='b',label='test')
plt.legend()
plt.show()
# 结果发现模型有轻微过拟合，且随着max_depth增大而增大，最终选择max_depth = 3,并更改cirterion = 'entropy'