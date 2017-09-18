# -*- coding: utf-8 -*-
#kmeans 对于蛇形的的相互交替的不是很好，只能是圆心的簇
#dbscan为密度算法密度越大越好。
#第一步是以一个点为核心，一个值为半径，再一个值为半径内要多少个点
#第二步如果在这半径中有要求的值就为一类，不是的就不为一类，
#不为一类的从新选点重复上面的
#为一类的在圆中选点再进行，知道不能找到任何一个类为止。
#交集的为一个大类。

import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
data=pandas.read_csv("D:\\bigdata\\7.2\\data.csv")
#设定半径
eps=0.2
#最少的点的个数
minpts=5
model=DBSCAN(eps,minpts)
model.fit(data)
#没有predict。只能下面的形式

data["type"]=model.fit_predict(data)

plt.scatter(
            data["x"],
            data["y"],
            c=data["type"]
            )