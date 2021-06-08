from sklearn import tree
from sklearn.datasets import load_wine
# 导入训练集和测试集的类
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz
import matplotlib.pyplot as plt


wine = load_wine()

# 用pandas先观察数据结构
# data = pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1)

# 输入数据，标签，指定30%的数据用于测试集，剩下70%用作训练集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data,wine.target,test_size=0.3)

# sklearn建模分三步。第一步实例化。criterion表示选择哪种决策树算法，random_state传入随机性种子，能控制随机性。
# splitter是控制随机性，默认是best。这些参数最终都要以score得分最高为依据来进行选择
test = []
for i in range(1,11,1):
    clf = tree.DecisionTreeClassifier(criterion='entropy',random_state=30,splitter='random',max_depth=i)
    clf = clf.fit(Xtrain,Ytrain)
    score = clf.score(Xtest,Ytest) # 返回预测的准确度accuracy
    test.append(score)
# 绘制不同max_depth条件下模型的预测得分
plt.plot(range(1,11,1),test,color='r')
plt.xticks(range(1,11,1))
plt.xlabel('max_depth')
plt.ylabel('Score')
plt.grid()
plt.savefig('1.model_score.png')
plt.show()

#结果发现max_depth=3模型得分最高，所以选取max_depth=3来进行训练
clf_best = tree.DecisionTreeClassifier(criterion='entropy',random_state=30,splitter='random',max_depth=3)
clf_best = clf_best.fit(Xtrain,Ytrain)

# filled = True是个决策树填充颜色，为不同类配备相同的颜色。rounded是将决策树边角椭圆化
dot_data = tree.export_graphviz(clf_best,feature_names=wine.feature_names,class_names=['Qinjiu','sherry','belmode'],filled=True,rounded=True)
graph = graphviz.Source(dot_data)

graph.render('1.wine')