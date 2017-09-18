# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 10:12:24 2017

@author: TH
"""
import string
import jieba
import numpy
import codecs
import pandas
file=codecs.open("D:\\bigdata\\2.5\\红楼梦.txt","r","utf8")
content=file.read()
#停用词不能识别换行符所以要先去除
content=content.replace("\r\n"," ")
file.close()
#加入自己定义的要分成的词
jieba.load_userdict("D:\\bigdata\\2.5\\红楼梦.txt")
segments=[]
#segs是一个对象，由于内存空间有限，只先读取一部分。
stopwords=pandas.read_csv("D:\\bigdata\\2.5\\StopwordsCN.txt",index_col=False,quoting=3,sep="\t", encoding='utf8')
segs=jieba.cut(content)
for seg in segs:
      if len(seg)>1:
            seg=str(seg).replace("\r\n"," ")
           
            segments.append(seg);
segmentdf=pandas.DataFrame({"segment":segments})
stopwords=pandas.read_csv("D:\\bigdata\\2.5\\StopwordsCN.txt",index_col=False,quoting=3,sep="\t", encoding='utf-8')
segmentdf23=segmentdf[~(segmentdf.segment.isin(stopwords.stopword))]


mystopwords=pandas.Series([# 42 个文言虚词 
  '之', '其', '或', '亦', '方', '于', '即', '皆', '因', '仍', '故', 
  '尚', '呢', '了', '的', '着', '一', '不', '乃', '呀', '吗', '咧', 
  '啊', '把', '让', '向', '往', '是', '在', '越', '再', '更', '比', 
  '很', '偏', '别', '好', '可', '便', '就', '但', '儿', 
  # 高频副词 
  '又', '也', '都', '要', 
  # 高频代词 
  '这', '那', '你', '我', '他',
  #高频动词
  '来', '去', '道', '笑', '说',
  #空格
  ' ', ''])
e=segmentdf[~segmentdf.segment.isin(mystopwords)]
segstat=segmentdf.groupby(by=["segment"])["segment"].agg({"计数":numpy.size}).reset_index().sort_values(by=["计数"],ascending=False)
segstat.head(100)


#进行绘图
from wordcloud import WordCloud,ImageColorGenerator
import matplotlib.pyplot as plt
wordcloud=WordCloud(font_path="D:\\bigdata\\2.5\\simhei.ttf",background_color="black")
words=segstat.set_index("segment").to_dict()
wordcloud=wordcloud.fit_words(words["计数"])
plt.imshow(wordcloud)


#修改绘图模式
from scipy.misc import imread
#图片读成数组
bimg=imread("D:\\bigdata\\2.5\\贾宝玉2.png")
wordcloud=WordCloud(background_color="white",font_path="D:\\bigdata\\2.5\\simhei.ttf",mask=bimg)
wordcloud=wordcloud.fit_words(words["计数"])

plt.figure(
    num=None, 
    figsize=(8, 6), dpi=80, 
    facecolor='w', edgecolor='k'
)
#读取图片的颜色用于设置字体的样色
bimgcolors=ImageColorGenerator(bimg)
plt.axis("off")
plt.imshow(wordcloud.recolor(color_func=bimgcolors))
plt.show()
