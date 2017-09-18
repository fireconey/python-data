# -*- coding: utf-8 -*-
contents = [
    '我 是 中国 人。',
    '你 是 美国 人。',
    '他 叫 什么 名字？',
    '她 是 谁 啊？'
];

from sklearn.feature_extraction.text import CountVectorizer

countVectorizer = CountVectorizer()
#数据矩阵
textVector = countVectorizer.fit_transform(contents);


textVector.todense()
countVectorizer.vocabulary_

#mid_df=0表示所有的都不省略
countVectorizer = CountVectorizer(
    min_df=0, 
    token_pattern=r"\b\w+\b"
)
textVector = countVectorizer.fit_transform(contents);
print(textVector)

textVector.todense()
countVectorizer.vocabulary_

from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer()
#这里的热度点使用坐标值和气对应的
tfidf = transformer.fit_transform(textVector)
import pandas;
#变成矩阵后依然列明使用0~12表示的
TFIDFDataFrame = pandas.DataFrame(tfidf.toarray());
#列明变成对应的对应的字符
TFIDFDataFrame.columns = countVectorizer.get_feature_names();


import numpy;
#这里是使用小标数进行表示的如最大的为9号坐标，则9排列最后。
TFIDFSorted = numpy.argsort(tfidf.toarray(), axis=1)[:,:]
#
print(TFIDFDataFrame.columns[TFIDFSorted].values)
