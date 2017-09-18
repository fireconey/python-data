# -*- coding: utf-8 -*-
import numpy
import pandas 
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
data=pandas.read_csv("D:\\bigdata\\4.1\\data.csv")


#画散点图
plt.scatter(data.广告投入,data.销售额)
#计算相关性
data.corr()

#估计模型参数，建立回归模型
x=data[["广告投入"]]
y=data[["销售额"]]
lrModel=LinearRegression()
#模型的训练
lrModel.fit(x,y)
#进行评分（就是相关系数R）
lrModel.score(x,y)


#进行预测(可以进行多个属性多组的预测所以是二维数组)
predict=lrModel.predict([[50]])
#截距(也就是参数A)0表示第一行的A
alpha=lrModel.intercept_[0]

#参数b【0】【0】表示第一行第一列的b--由于预测的
beta=lrModel.coef_[0][0]
beta
