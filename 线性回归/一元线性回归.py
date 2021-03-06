# 线性回归
import numpy as np  # 快速操作结构数组的工具
import matplotlib.pyplot as plt  # 可视化绘制
from sklearn.linear_model import LinearRegression  # 线性回归
from sklearn.metrics import r2_score

# 样本数据集，第一列为x，第二列为y，在x和y之间建立回归模型
data = [
    [0.067732, 3.176513], [0.427810, 3.816464], [0.995731, 4.550095], [0.738336, 4.256571], [0.981083, 4.560815],
    [0.526171, 3.929515], [0.378887, 3.526170], [0.033859, 3.156393], [0.132791, 3.110301], [0.138306, 3.149813],
    [0.247809, 3.476346], [0.648270, 4.119688], [0.731209, 4.282233], [0.236833, 3.486582], [0.969788, 4.655492],
    [0.607492, 3.965162], [0.358622, 3.514900], [0.147846, 3.125947], [0.637820, 4.094115], [0.230372, 3.476039],
    [0.070237, 3.210610], [0.067154, 3.190612], [0.925577, 4.631504], [0.717733, 4.295890], [0.015371, 3.085028],
    [0.335070, 3.448080], [0.040486, 3.167440], [0.212575, 3.364266], [0.617218, 3.993482], [0.541196, 3.891471]
]

# 生成X和y矩阵
dataMat = np.array(data)
print(dataMat)
X = dataMat[:, 0:1]  # 变量x
y = dataMat[:, 1]  # 变量y

# ========线性回归========
model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model.fit(X, y)  # 线性回归建模
print('回归系数:\n', model.coef_)
print('截距:\n', model.intercept_)
print('线性回归模型:\n', model)
# 使用模型预测
predicted = model.predict(X)

print('R2:',r2_score(y,predicted))
# 绘制散点图 参数：x横轴 y纵轴
plt.scatter(X, y, marker='x')
plt.plot(X, predicted, c='r')

# 绘制x轴和y轴坐标
plt.xlabel("x")
plt.ylabel("y")

# 显示图形
plt.savefig('一元线性回归.png',dpi=300)
plt.show()