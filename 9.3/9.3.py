
#时间序列预测，就是通过已知的，找出规律（和时间相关），预测未来的
#不仅时间其他也可以预测
#统计特征（方差，均值等）不随时间变化，才可能预测
#如果变动则使用差分手段
#差分：D=f（X（n+1））-f（X（n））
#AR模型  以前所有的值决定后面的 x=c+求和（q*X(小标t-i)+E（下表t）） i从1开始
#MA模型 移动的一定数决定值x=u+求和（B*E(小标t-i)+E（下表t）） i从1开始
import pandas as pd;
import statsmodels.api as sm;
import matplotlib.pyplot as plt;
import statsmodels.tsa.stattools as ts;

#
dp=lambda x:pd.datetime.strptime(x,"%Y%m%d")
#parse_dates 表示后面指定的参数列要识别为时间
data=pd.read_csv("D:\\bigdata\\9.3\\date.csv",
                 parse_dates=["date"],
#                 把时间序列变成那种格式的时间表示法
                 date_parser=dp,
#                 使用什么来当索引
                 index_col="date"
                 )


plt.figure(figsize=(10,6))
#后面的raw表示曲线叫啥
plt.plot(data,"r",label="Raw")
#叫啥曲线的图例放在那里，0表示在自动排版
plt.legend(loc=0)
#自己定义一个格式转换的函数便宜评分的查看形式。
def tagadf(t):
    result=pd.DataFrame(index=[
            "Test Statistic Value", "p-value", "Lags Used",
            "Number of Observations Used",
            "Critical Value(1%)", "Critical Value(5%)", "Critical Value(10%)"
        ], columns=['value']
      );
    result['value']['Test Statistic Value'] = t[0]
    result['value']['p-value'] = t[1]
    result['value']['Lags Used'] = t[2]
    result['value']['Number of Observations Used'] = t[3]
    result['value']['Critical Value(1%)'] = t[4]['1%']
    result['value']['Critical Value(5%)'] = t[4]['5%']
    result['value']['Critical Value(10%)'] = t[4]['10%']
    return result
#loc 索整型索引
#loc 索引字符串索引
#ix 是 iloc 和 loc的合体
#adfuller表示AR模型训练出评估的指标
adfdata=ts.adfuller(data.iloc[:,0])

tagadf(adfdata)
#diff（1）表示一阶差分，然后得到一个列表，dropna（）表示删除空的
diff=data.diff(1).dropna()
#差分后看看数字是否稳定了
plt.figure(figsize=(10,6))
plt.plot(diff,"r",label="Diff")
plt.legend(loc=0)
#再使用AR进行训练出评估的指标
adfdiff=ts.adfuller(diff.iloc[:,0])
#使用自己定义的tagadf函数看看 其中的衡量好坏的标准
tagadf(adfdiff)

#使用模型进行拟合了,max_ar设定p值，max_ma设定q值
#参数ic为求解的准则
#AIC =  -2 ln(L) + 2 k 中文名字：赤池信息量 akaike information criterion
# BIC =  -2 ln(L) + ln(n)*k 中文名字：贝叶斯信息量 bayesian information criterion
# HQ =  -2 ln(L) + ln(ln(n))*k hannan-quinn criterion
#ic=sm.tsa.arma_order_select_ic(
#          diff,
#          max_ar=20,
#          max_ma=20,
#          ic="aic")
#ic(左边的)是求出的p，q组合
order=(15,9)

#根据已知的p,q 进行训练
arm=sm.tsa.ARMA(diff,order).fit()

#拟合出来的和原有的进行差值计算
delta=arm.fittedvalues-diff.iloc[:,0]
#.var()表示方差 。下面是模型的评分
score=1-delta.var()/diff.var()

plt.figure(figsize=(10,6))
plt.plot(diff,"r",label="Raw")
#画出训练后的值
plt.plot(arm.fittedvalues,"g",label='ARMA Model')
plt.legend()

#预测（是差分后的值，需要根据差分的公式继续还原）
p=arm.predict(
          start="2016-03-31",
          end="2016-04-10"
          )
#一个*表示这个参数可变（长短），。两个*表示如果是不是标准的参数，就
#放在字典中，使用字典表示。下面的函数是还原
#这里的预测后的数据是后面向前的差分数列 所以要还原，同时要指定第一个数是多少
def revert(diffValues, *lastvalues):
     for i in range(len(lastvalues)):
          t.append(lastvalues[i])
          result=[]
          lv=lastvalues[i]
          for dv in diffValues:
               lv=dv+lv
               result.append(lv)
          diffValues=result
     return diffValues;

#1035表示最后一行的数字值
r=revert(p,10395)












