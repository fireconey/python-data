# -*- coding: utf-8 -*-
import pandas
data=pandas.read_csv("D:\\bigdata\\5.3\\data.csv")
dummycolumns=["Gender", "ParentEncouragement"]
for colum in dummycolumns:
      data[colum]=data[colum].astype("category")

dummydata=pandas.get_dummies(
            data,
            columns=dummycolumns,
            prefix_sep="=",
            drop_first=True
    )

dummydata.head()
dummydata.columns

fdata=dummydata[[
    'ParentIncome', 'IQ', 'Gender=Male',
    'ParentEncouragement=Not Encouraged'
]]

tdata=dummydata["CollegePlans"]

from sklearn.tree import DecisionTreeClassifier  as dc
dt=dc(max_leaf_nodes=8)

dt.fit(fdata,tdata)


#要求安装graphviz.exe 且配置了path（bin）
from sklearn.tree import export_graphviz
with open('D:\\bigdata\\5.3\\data.dot','w') as f:
    f = export_graphviz(dt, out_file=f)
    
    
    
import pydot 
from sklearn.externals.six import StringIO

dot_data=StringIO()
export_graphviz(
      dt,
      out_file=dot_data,
      class_names=["不计划", "计划"],
       feature_names=["父母收入", "智商", "性别=男", "父母鼓励=不鼓励"],
    filled=True, rounded=True, special_characters=True
      )

graph=pydot.graph_from_dot_data(dot_data.getvalue())
graph.get_node("node")[0].set_fontname("Microsoft YaHei")

graph.write_png('D:\\bigdata\\5.3\\tree.png')
