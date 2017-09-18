# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 15:54:31 2017

@author: TH
"""
#协同过滤，先找到相似的商品，
#找到相似的商品，
#商品排序再排除另一个客户已经有的
#每个用户对商品评分就构成  一个用户向量，没购买的就是0
#T0(1,0,3,0,0,0,5)
#T1(0,4,5,1,0,0,0)
#算欧拉距离，越小越相似，为了使用0~1表示相似性
#进行了变形 sim（x，y）=1/(1+欧拉距离)  ，1最相似。
#1、选K个邻居进行计算，不论多远，取理他近的
#2、画个半径，在半径内的邻居
import numpy as np
import pandas as pd
#欧拉距离模块
from sklearn.metrics.pairwise import euclidean_distances as eu
data=pd.read_csv("D:\\bigdata\\8.2\\data.csv")
#生成用户评分矩阵
ur=data.pivot_table(
          index="ItemID",
          columns="UserID",
          aggfunc=sum,
          fill_value=0
          )

ur.columns=ur.columns.droplevel(0)
del ur.columns.name


dist=pd.DataFrame(eu(ur))
dist.index=ur.index
dist.columns=ur.index
#计数商品的相似度
sim=1/(1+dist)
#为用户为3的推荐
userid=3
#取出用户3 的数据
userit=ur[3]
uo=userit
#dot是T乘法

score=pd.DataFrame(
          np.dot(sim,userit)
       )
result=ur.index[score.sort_values(0,ascending=False).index.values]
#result  为推荐商品的id值
result
