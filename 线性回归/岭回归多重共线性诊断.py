# 导入计算膨胀因子的库
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
# get_loc(i) 返回对应列名所在的索引

data = pd.read_excel('岭回归数据.xlsx')

x= data

# vif>10则说明存在多重共线性，需要处理
vif=[variance_inflation_factor(x.values,x.columns.get_loc(i)) for i in x.columns]

# 测试x.columns.get_loc(i)的含义，这里为获得不同列名的索引值
for i in x.columns:
    print(x.columns.get_loc(i))

print(vif)