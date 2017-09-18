# -*- coding: utf-8 -*-
import numpy
X = [
    12.5, 15.3, 23.2, 26.4, 33.5, 
    34.4, 39.4, 45.2, 55.4, 60.9
]
Y = [
    21.2, 23.9, 32.9, 34.1, 42.5, 
    43.2, 49.0, 52.8, 59.4, 63.5
]
#均值
Xmean=numpy.mean(X)
Ymean=numpy.mean(Y)

#标准差
xsd=numpy.std(X)
ysd=numpy.std(Y)

#Z分数
zx=(X-Xmean)/xsd
zy=(Y-Ymean)/ysd

#相关系数
r=numpy.sum(zx*zy)/len(X)
r
#直接调用函数也可以计算
numpy.corrcoef(X,Y)
#直接调用函数也可以计算使用pandas
import pandas
data=pandas.DataFrame({"X":X,"Y":Y})
#直接算两两相关性
data.corr()
