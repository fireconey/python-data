# -*- coding: utf-8 -*-
#相关算法
#支撑度为，满足条件的在样本中的比例
#自信度为在满足条件的在一个前提条例样本（如买啤酒与没有买啤酒中前提是买啤酒的。）中的占比
#提升度是自信度除以支撑度
import pandas as pd
from apyori import apriori
data=pd.read_csv("D:\\bigdata\\8.1\\data.csv")
#由于使用pandas打开的数据apriori不兼容所有要数据转换
#按照同一个人进行了分组
transaction=data.groupby(by="交易ID").apply(
            lambda x:list(x.购买商品)).values
transaction

#进行计算,默认要出现次数为3，没有出现的就剔除了。所以有可乐啤酒，没有啤酒可乐。
result=list(apriori(transaction,max_length=100))
result

sup=[]
cof=[]
lif=[]

#基于项的
bases=[]
#基于推导的
adds=[]

#由于数据结构是这样的，其中只有一个一维数组。
#RelationRecord(items=frozenset({'可乐'}), 
#               support=0.4,
#               ordered_statistics=[OrderedStatistic(items_base=frozenset(),
#                                                    items_add=frozenset({'可乐'}),
#                                                    confidence=0.4,
#                                                    lift=1.
#						                                           )
#                                   ]
#               )

for r in result:
      sup.append(r.support)
      cof.append(r.ordered_statistics[0].confidence)
      lif.append(r.ordered_statistics[0].lift)
      bases.append(list(r.ordered_statistics[0].items_base))
      adds.append(list(r.ordered_statistics[0].items_add))
     

result=pd.DataFrame({
            "sup":sup,
            "cof":cof,
            "lif":lif,
            "base":bases,
            "add":adds
            })
      
r=result[(result.lif>1) & (result.sup>0.5) & (result.cof>0.5)]