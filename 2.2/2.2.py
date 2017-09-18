#coding:utf-8
import jieba
import os
import os.path
import codecs
import sys
sys.setdefaultencoding('gbk')
filepaths=[]
filecontents=[]
for root ,dirs,files in os.walk('D:\\bigdata\\2.1\\SogouC.mini\\Sample'):
    for name in files:
        filepath=os.path.join(root,name)
        filepaths.append(filepath)
        f=codecs.open(filepath,"r","utf-8")
        filecontent=f.read()
        f.close()
        filecontents.append(filecontent)
        
#列表的名称一定要不和变量一样如果filepath:filepath就会出错。
import pandas;
corpos = pandas.DataFrame({"filepath": filepaths,
                          "filecontent":filecontents})
segments=[]                       
filepaths=[]
for index , row in corpos.iterrows():
    filecontent=row["filecontent"]
    filepath=row["filepath"]
    segs=jieba.cut(filecontent)
    for seg in segs:
        segments.append(seg)
        filepaths.append(filepath)
segmentd=pandas.DataFrame({"segment":segments,
                           "filepath1":filepaths})


#词频统计
import numpy
segstat=segmentd.groupby(by=["segment"])["segment"].agg({"计数":numpy.size}).reset_index().sort_values(by=["计数"],ascending=False);


   