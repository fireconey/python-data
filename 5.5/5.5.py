# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 20:05:58 2017

@author: TH
"""

#支撑向量机=一刀分两类=可以弯曲。

import scipy.io as si
data=si.loadmat("d:\\bigdata\\5.5\\data.mat")
data["categories"]

fdata=data["wine"]
tdata=data["wine_labels"].reshape(-1)

from sklearn import svm as sv
from sklearn.model_selection import cross_val_score as sc
sm=sv.SVC()
sc(
   sm,
   fdata,tdata,cv=3
   )
sm=sv.NuSVC()
sc(
   sm,fdata,tdata,cv=3)
sm=sv.LinearSVC()
sc(
   sm,fdata,tdata,cv=3)
sm.fit(fdata,tdata)
sm.score(fdata,tdata)
sm.predict(fdata)



