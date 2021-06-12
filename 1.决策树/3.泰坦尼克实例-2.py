# 网格搜索：能够帮助我们同时调整多个参数的技术，枚举技术
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
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

# parameters是一串参数和这些参数对应的我们希望网格搜索来搜索的参数取值范围
parameters = {'criterion':['gini','entropy'],
             'splitter':['best',"random"],
             'max_depth':[*range(1,10)],
              'min_samples_leaf':[*range(1,50,5)],
              'min_impurity_decrease':np.linspace(0,0.5,5)
              }
clf = DecisionTreeClassifier(random_state=25)
GS = GridSearchCV(clf,parameters,cv=10)
GS = GS.fit(Xtrain,Ytrain)

# 打印最佳组合及得分
print(GS.best_params_)
print(GS.best_score_)

'''
{'criterion': 'entropy', 'max_depth': 6, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'splitter': 'best'}
0.8264464925755248
'''