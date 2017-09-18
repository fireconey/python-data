# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 18:59:46 2017

@author: TH
"""

#随机深林 就是多个决策数，然后选择最优的
#决策数id3 离散 c4算法是连续算法。
#随机森林可以用于大量的算法。
#决策数进行参数调优，也可以达到随机森林的准确度。（调节叶子节点数）
import pandas as pd
data=pd.read_csv("D:\\bigdata\\5.4\\data.csv")
dummycolumns=["Gender", "ParentEncouragement"]
for column in dummycolumns:
      data[column]=data[column].astype("category")
dummiesdata=pd.get_dummies(
            data,
            columns=dummycolumns,
            prefix=dummycolumns,
            prefix_sep="=",
            drop_first=True
            )

fdata=dummiesdata[[
    'ParentIncome', 'IQ', 'Gender=Male',
    'ParentEncouragement=Not Encouraged']]

tdata=dummiesdata["CollegePlans"]

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

#进行决策树数交叉验证
dm=DecisionTreeClassifier()
dtc=cross_val_score(
            dm,
            fdata,
            tdata,
            cv=10)
dtc.mean(0)


#进行随机森林交叉验证
rf=RandomForestClassifier()
rfc=cross_val_score(
            rf,
            fdata,
            tdata,
            cv=10)
rfc.mean()

#进行了调用再验证
dm=DecisionTreeClassifier(max_leaf_nodes=8)
dtc=cross_val_score(
            dm,
            fdata,
            tdata,
            cv=10)
dtc.mean(0)
rf.fit(fdata,tdata)

#选择要参数化的列
duml=["Gender","ParentEncouragement"]
prdata=pd.DataFrame({
            "ParentIncome":[2000],
            "Gender":["Male"],
            "IQ":[100],
            "ParentEncouragement":["Encouraged"]
         })
#columns 表示要序列化的列，prefix表示要全面添加的前缀
#      得到的是没序列化的和序列化的
inputdata=pd.get_dummies(
            prdata,
            columns=duml,
            prefix=duml,
            prefix_sep="=",
           
            )

rf.predict(inputdata)