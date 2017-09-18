# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 13:20:07 2017

@author: TH
"""
#层次聚类就是先找到两个连之间的最少距离的点两个点，
#这两个点分为一类，两个点中点看为另外一个点再和其他的算最短距离
#一次重复上面的算法
#直到所有的点都归到要求的类的数量
#会形成一个决策数每个枝有两片叶子。

import pandas as pd  
import  matplotlib.pyplot as plt
from sklearn.decomposition  import PCA
import scipy.cluster.hierarchy as hclu
data=pd.read_csv("D:\\bigdata\\7.1\\data.csv")
fcolumns=[
    '工作日上班时电话时长', '工作日下半时电话时长', 
    '周末电话时长', '国际电话时长', '平均每次通话时长'
]
#centeroid 两点间中间距离算法
#single表示两点间最短距离
#参数complet表示两点间最长距离
#下面是对进行了分类，但是没有进行标注，是一个决策数的数据(坐标)，每支直到只有两个叶子。
linkage=hclu.linkage(data[fcolumns],
                     method="centroid")
#下面是绘图
hclu.dendrogram(linkage)
plt.savefig('plot_dendrogram.png')  
hclu.dendrogram(
            linkage,
            truncate_mode="lastp",
#            表示要有多少个叶子
            p=8,
            leaf_font_size=12)

#进行类别的标注 ，选决策树的叶子数是规定的那层（减枝处理）
#参数criterion=maxclust表示最大划分的方法
ptar=hclu.fcluster(
            linkage,
#            表示分类的最大个数
            3,
            criterion="maxclust"
            )

#降维
pca2=PCA(n_components=2)
data2=pd.DataFrame(pca2.fit_transform(data[fcolumns]))

plt.scatter(
            data2[0],
            data2[1],
            c=ptar)



#下面所有为从新组织数据
dMean = pd.DataFrame(columns=fcolumns+['分类'])
data_gb = data[fcolumns].groupby(ptar)
i = 0;
for g in data_gb.groups:
    rMean = data_gb.get_group(g).mean()
    rMean['分类'] = g;
    dMean = dMean.append(rMean, ignore_index=True)
    subData = data_gb.get_group(g)
    for column in fcolumns:
        i = i+1;
        p = plt.subplot(3, 5, i)
        p.set_title(column)
        p.set_ylabel(str(g) + "分类")
        plt.hist(subData[column], bins=20)