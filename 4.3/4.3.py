# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 11:10:50 2017

@author: TH
"""
#一元非线性y=ax^n+a1x^n-1.....
#变成y=ax1+a1x1.......
import numpy as np
import pandas as mp
from sklearn.linear_model import LinearRegression as ln
import matplotlib as plt
data=mp.read_csv("D:\\bigdata\\4.3\\data.csv")
x=data[["等级"]]
y=data[["资源"]]
font={"family":"SimHei"}
plt.rc("font",**font)

plt.rcParams['axes.unicode_minus'] = False

from pandas.tools.plotting import scatter_matrix as mtr
mtr(data[["等级","资源"]],alpha=0.8,figsize=(10,10),diagonal="kde")

from sklearn.preprocessing import PolynomialFeatures as pl
pf=pl(degree=100)
x1=pf.fit_transform(x)
lr=ln()
lr.fit(x1,y)
lr.score(x1,y)

#由于多元非线性的输入数变成了一元的数来计数的，所以预测的也要转换
tr=pf.fit_transform([[21],[22],[23]])
lr.predict(tr)
