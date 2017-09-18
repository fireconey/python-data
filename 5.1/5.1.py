# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 14:48:09 2017

@author: TH
"""
#分类预测问题
#监督学习：有标注结果的数据先训后，对数据进行标注，看正确率
#监督就是有y的函数。无监督的就是聚类等
#----*KNN算法*---------就是也相近越相似。
#数据分为k份，先第一份为测试集其他为训练集，再第二份为测试集。。。。。。
#可以有k个得分。平均分就是判断标准。


import numpy
from sklearn import datasets
ir=datasets.load_iris()

#查看数据的规模
ir.data.shape

#查看训练的目标的总类
numpy.unique(ir.target)

from sklearn.model_selection import train_test_split

data_train,data_test,target_train,target_test=train_test_split(
          ir.data,
          ir.target,
#          所有的数据按照训练和测试的3:7分
          test_size=0.3
            )

data_train.shape
data_test.shape
target_test.shape
target_test.shape

#使用knn方法进行数据的训练
from sklearn import neighbors as nb
#n_neighbors 表示训练数据分为几类开始训练。这里分为3份，其中一份用于验证，不交叉（交叉数这顶1）。
k=nb.KNeighborsClassifier(n_neighbors=5)
k.fit(data_train,target_train)
k.score(data_test,target_test)

#设定交叉多少则验证 这里为5则
from sklearn.model_selection import cross_val_score

cross_val_score(k,ir.data,ir.target,cv=5)

#预测
k.predict([[6.2,4,4,2]])
