# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 12:34:59 2017

@author: TH
"""
#逻辑回归是用于分类，小于0.5不是此类
import pandas as pd
data=pd.read_csv("D:\\bigdata\\4.4\\data.csv")
##删除无效的行
data=data.dropna()
data.shape

#定义要抽取的列
dummycolumns=[
    'Gender', 'Home Ownership', 
    'Internet Connection', 'Marital Status',
    'Movie Selector', 'Prerec Format', 'TV Signal'
]
#由于先前的值就是个字符，不代表什么东西，
#现在告诉计数机它表示一类,能够知道有多少种，此不是必须写。
for column in dummycolumns:
      data[column]=data[column].astype("category")

#dummycolumns参数表示要虚拟化的列，列名为列中的所有值，值为在没有选择
#      列名情况下的是1或0
# l1  l2
# t   a
# y   b  
      
#l1  l2a  l2b
#t    1     0
#y    0     1
dummiesdata=pd.get_dummies(
            data,
#            要处理哪些列
            columns=dummycolumns,
#            如果不同列的值有相同的则在值前添加前缀
            prefix=dummycolumns,
#            前缀和值得连接方式
            prefix_sep=" ",
#            从备选中删除一个避免共轭
            drop_first=True
            )     
#unique()是去重，看有多少类
data.Gender.unique()
dummiesdata.columns

#要知道1表示什么可以看看列名的后缀




#对于有大小的虚拟数据的处理
#下列的是说明的作用没起啥意思
"""
博士后    Post-Doc
博士      Doctorate
硕士      Master's Degree
学士      Bachelor's Degree
副学士    Associate's Degree
专业院校  Some College
职业学校  Trade School
高中      High School
小学      Grade School
"""
#定正式义级别
edu={
     'Post-Doc': 9,
    'Doctorate': 8,
    'Master\'s Degree': 7,
    'Bachelor\'s Degree': 6,
    'Associate\'s Degree': 5,
    'Some College': 4,
    'Trade School': 3,
    'High School': 2,
    'Grade School': 1
     }
#加了一列叫Education Level Map  ，是影射出来的值
dummiesdata["Education Level Map"]=dummiesdata['Education Level'].map(edu)

#再定义不同的值
freqmap={
        'Never': 0,
    'Rarely': 1,
    'Monthly': 2,
    'Weekly': 3,
    'Daily': 4
         }
#再次在后面添加列
dummiesdata['PPV Freq Map'] = dummiesdata['PPV Freq'].map(freqmap)
dummiesdata['Theater Freq Map'] = dummiesdata['Theater Freq'].map(freqmap)
dummiesdata['TV Movie Freq Map'] = dummiesdata['TV Movie Freq'].map(freqmap)
dummiesdata['Prerec Buying Freq Map'] = dummiesdata['Prerec Buying Freq'].map(freqmap)
dummiesdata['Prerec Renting Freq Map'] = dummiesdata['Prerec Renting Freq'].map(freqmap)
dummiesdata['Prerec Viewing Freq Map'] = dummiesdata['Prerec Viewing Freq'].map(freqmap)

#看看有哪些列明了
dummiesdata.columns

#要选取的列明

dummiesselect = [
    'Age', 'Num Bathrooms', 'Num Bedrooms', 'Num Cars', 'Num Children', 'Num TVs', 
    'Education Level Map', 'PPV Freq Map', 'Theater Freq Map', 'TV Movie Freq Map', 
    'Prerec Buying Freq Map', 'Prerec Renting Freq Map', 'Prerec Viewing Freq Map', 
    'Gender Male',
    'Internet Connection DSL', 'Internet Connection Dial-Up', 
    'Internet Connection IDSN', 'Internet Connection No Internet Connection',
    'Internet Connection Other', 
    'Marital Status Married', 'Marital Status Never Married', 
    'Marital Status Other', 'Marital Status Separated', 
    'Movie Selector Me', 'Movie Selector Other', 'Movie Selector Spouse/Partner', 
    'Prerec Format DVD', 'Prerec Format Laserdisk', 'Prerec Format Other', 
    'Prerec Format VHS', 'Prerec Format Video CD', 
    'TV Signal Analog antennae', 'TV Signal Cable', 
    'TV Signal Digital Satellite', 'TV Signal Don\'t watch TV'
]
#dummiesdata是处理两类之后的数据结构，不是先前形成的。
inputdata=dummiesdata[dummiesselect]

outputdata=dummiesdata[['Home Ownership Rent']]

from sklearn import linear_model as ln
#regression 回归
lr=ln.LogisticRegression()

lr.fit(inputdata,outputdata)

lr.score(inputdata,outputdata)





#上面的是数据的训练，这里是预测


newData =pd.read_csv(
    'D:\\bigdata\\4.4\\newData.csv', 
    encoding='utf8'
)

for column in dummycolumns:
    newData[column] = newData[column].astype(
        'category', 
        categories=data[column].cat.categories
    )

newData = newData.dropna()

newData['Education Level Map'] = newData['Education Level'].map(edu)
newData['PPV Freq Map'] = newData['PPV Freq'].map(freqmap)
newData['Theater Freq Map'] = newData['Theater Freq'].map(freqmap)
newData['TV Movie Freq Map'] = newData['TV Movie Freq'].map(freqmap)
newData['Prerec Buying Freq Map'] = newData['Prerec Buying Freq'].map(freqmap)
newData['Prerec Renting Freq Map'] = newData['Prerec Renting Freq'].map(freqmap)
newData['Prerec Viewing Freq Map'] = newData['Prerec Viewing Freq'].map(freqmap)

dummiesNewData = pd.get_dummies(
    newData, 
    columns=dummycolumns,
    prefix=dummycolumns,
    prefix_sep=" ",
    drop_first=True
)

inputNewData = dummiesNewData[dummiesselect]

ttt=lr.predict(inputNewData)
