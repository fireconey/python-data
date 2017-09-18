# -*- coding: utf-8 -*-

import os;
import numpy;
import os.path;
import codecs;

filePaths = [];
fileContents = [];
for root, dirs, files in os.walk(
    "D:\\bigdata\\2.9\\SogouC.mini\\Sample"
):
    for name in files:
        filePath = os.path.join(root, name);
        filePaths.append(filePath);
        f = codecs.open(filePath, 'r', 'utf-8')
        fileContent = f.read()
        f.close()
        fileContents.append(fileContent)

import pandas;
corpos = pandas.DataFrame({
    'filePath': filePaths, 
    'fileContent': fileContents
});


import re
#匹配中文的分词
zhPattern = re.compile(u'[\u4e00-\u9fa5]+')

import jieba

segments = []
filePaths = []
for index, row in corpos.iterrows():
    segments = []
    filePath = row['filePath']
    fileContent = row['fileContent']
    segs = jieba.cut(fileContent)    
    for seg in segs:
        if zhPattern.search(seg):
            segments.append(seg)
    filePaths.append(filePath)
    row['fileContent'] = " ".join(segments);

from sklearn.feature_extraction.text import CountVectorizer

stopwords = pandas.read_csv(
    "D:\\bigdata\\2.9\\StopwordsCN.txt", 
    encoding='utf8', 
    index_col=False,
    quoting=3,
    sep="\t"
)

countVectorizer = CountVectorizer(
    stop_words=list(stopwords['stopword'].values),
    min_df=0, token_pattern=r"\b\w+\b"
)
textVector = countVectorizer.fit_transform(
#一行是存储一个文本的分词的内容，且进行了空白的切割（使用空格连接）
    corpos['fileContent']
)

from sklearn.metrics import pairwise_distances
#第一行和第1行计算出的放在0,0。第一行和第二喊计算出的放在0,1
#显示的是角度
distance_matrix = pairwise_distances(
    textVector, 
    metric="cosine"
)

m = 1- pandas.DataFrame(distance_matrix)
m.columns = filePaths;
m.index = filePaths;
#从小到大排列，第一个为0是自己所以不取，使用1：开始
sort = numpy.argsort(distance_matrix, axis=1)[:, 1:6]
#根据排列找出文件的路径,上面的每一行值是列的值，每一行小面找到对应的行的值。
#然后按列排列。
similarity5 = pandas.Index(filePaths)[sort].values
similarityDF = pandas.DataFrame({
    'filePath':corpos.filePath, 
    's1': similarity5[:, 0], 
    's2': similarity5[:, 1], 
    's3': similarity5[:, 2], 
    's4': similarity5[:, 3], 
    's5': similarity5[:, 4]
})
