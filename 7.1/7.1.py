# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 20:59:37 2017

@author: TH
"""
#k_mean 就是自动把数据分为k类
#第一步是随机选k个点（质心）
#第二步：随机选一个点看里那个质心最近，就归为那个类。
#第三步就是算添加新点的那个类的新的质心（质心移动）
#重复上面的算法，最后质心不动了。
#
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data=pd.read_csv("D:\\bigdata\\7.1\\data.csv")
fcolumns=[
    '工作日上班时电话时长', '工作日下半时电话时长', 
    '周末电话时长', 
    '国际电话时长', '总电话时长', '平均每次通话时长'
]

import matplotlib;
from pandas.tools.plotting import scatter_matrix
font={"family":"Simhei"}

matplotlib.rc("font",**font)
matplotlib.rcParams["axes.unicode_minus"]=False
scatter_matrix(
            data[fcolumns],
#            图像的大小
            figsize=(10,10),
#            是柱形图还是曲线图
            diagonal="hist")
#计算列的两两相关性
dcorr=data[fcolumns].corr()
 
fcolumns = [
    '工作日上班时电话时长', '工作日下半时电话时长', 
    '周末电话时长', '国际电话时长', 
    '平均每次通话时长'
]
#绛位函数设定,里面的参数是维度
pca_2=PCA(n_components=3)

data_pca_2=pd.DataFrame(
#            导入数据进行降维
            pca_2.fit_transform(data[fcolumns])
            )

plt.scatter(       
     data_pca_2[0], 
     data_pca_2[1],   
            )

#要求分为三类
km=KMeans(n_clusters=3)
km=km.fit(data[fcolumns])

ptarget=km.predict(data[fcolumns])

pd.crosstab(ptarget,ptarget)

plt.scatter(
#            分三类是按照第一列分的，其余列分到不同的层中
#如下面参数为1,2就会出现乱图
#x坐标
   data_pca_2[0],
#   y坐标
    data_pca_2[2], 
    20,
# 第三个参数是表示点的大小    默认为20
#    下面是颜色,是根据分类依据来显示的
    c=ptarget
            )


#以下为统计各个类的差异
#定义一个空的列表，列明为fcolumns再增加一个“分类”
dm=pd.DataFrame(columns=fcolumns+["分类"])
#上面通关过kmeans算的了一个分类的数列(依据)ptarget这里就是分类
data_gb=data[fcolumns].groupby(ptarget)
i=0

plt.figure(figsize=(30,30))
#data_gb.groups是得到分组的标记
for g in data_gb.groups:
#      得到某一分类的平均值  data_gb.get_group(g) 得到分组的所有标记下的所有值。
      rm=data_gb.get_group(g).mean()
#      标记分类为g
      rm["分类"]=g;
#      数据田间appened是一列数据添加到一行中取
      dm=dm.append(rm,ignore_index=True)
      subd=data_gb.get_group(g)
      for column in fcolumns:
            i=i+1;
#            i是标记那第几个图。3行5列
            p=plt.subplot(3,5,i)
            p.set_title(column)
            p.set_ylabel(str(g)+"分类",fontsize=12)
            plt.hist(subd[column],bins=20)

