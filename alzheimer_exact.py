import pandas as pd
from scipy.stats import brunnermunzel

df1=pd.read_csv('data/alzheimer_tuned.csv',index_col='IDENTIFIER')
df2=pd.read_csv('data/alzheimer_label.csv')




df1=df1.transpose()
#transposeするときのindexを指定して転置する
df1['IDENTIFIER']=df1.index.copy()
#Nameがindexになっていて認識されないので新たにindexをコピーしたName列を作ってmergeする
df=pd.merge(df1,df2,on='IDENTIFIER',how='inner')
df=df.set_index('IDENTIFIER')
#IDENTIFIERをindexに指定した
df.to_csv('data/alzheimer_full.csv')

df=df.transpose()
#検定にかけるためにまた転置した

def Brunnermunzel(row):
    row['w'],row['p_value']=brunnermunzel(row[:34],row[34:])#distributionでｐ値を取得する方法を変更する、デフォはt分布でオプションで標準正規分布
    return row
df=df.apply(Brunnermunzel,axis=1)

df=df.query('p_value<0.05')
#p_valueが0.05未満の物だけを抽出した
df=df.drop(columns=['w'])


df=df.drop(columns=['p_value'])
df=df.transpose()

df=pd.merge(df,df2,on='IDENTIFIER',how='inner')
df.to_csv('data/alzheimer_test.csv',index=False)
