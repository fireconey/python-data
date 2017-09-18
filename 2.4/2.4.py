# -*- coding: utf-8 -*-
import jieba
import os
import os.path
import codecs
filepaths=[]
filecontents=[]
for root ,dirs,files in os.walk('D:\\bigdata\\2.1\\SogouC.mini\\Sample\\C000007'):
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


#导入要停用的词的文件
stopwords=pandas.read_csv("D:\\bigdata\\2.3\\StopwordsCN.txt",encoding="utf-8",index_col=False)


#移除停用词
fsegstat=segstat[~segstat.segment.isin(stopwords.stopword)]

#
##词云图片绘画词云
##http://www.lfd.uci.edu/~gohlke/pythonlibs/

#绘画词云
#http://www.lfd.uci.edu/~gohlke/pythonlibs/
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(
    font_path='D:\\bigdata\\2.4\\simhei.ttf', 
    background_color="black"
)

words =fsegstat.set_index("segment").to_dict()

wordcloud.fit_words(words['计数'])

plt.imshow(wordcloud)
plt.aixs("off")
plt.show()
plt.close()