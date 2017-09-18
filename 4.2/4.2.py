# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib  as mp
from sklearn.linear_model import LinearRegression  as ln
from pandas.tools.plotting import scatter_matrix as sc
#判定系数R进行了修改，考虑到了样本的个数和变量的个数
data=pd.read_csv("D:\\bigdata\\4.2\\data.csv")
font={"family":"SimHei"}
mp.rc("font",**font)

#四方图
sc(data[["店铺的面积","距离最近的车站","月营业额"]],
               figsize=(10,10) ,diagonal="kde")

#进行训练
x=data[["店铺的面积","距离最近的车站"]]
y=data[["月营业额"]]

lr=ln()
lr.fit(x,y)
lr.score(x,y)
#预测
lr.predict([[10,110],[4,200]])