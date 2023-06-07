import pandas as pd
import numpy as np

df=pd.read_csv('merged.csv',index_col='gene_name')
df1=df.iloc[:,:18]#dfのなかで1列目から18列目までの要素を取得した
df2=df.iloc[:,18:]#dfの中で19列目から最後までを取り出す(Ips)の方

def SetNorm1(row):
    row['CCALD_norm']=np.linalg.norm(row)
    return row
df1=df1.apply(SetNorm1,axis=1)

def SetNorm2(row):
    row['IPS_norm']=np.linalg.norm(row)
    return row
df2=df2.apply(SetNorm2,axis=1)#axisで指定した一行ずつnormを計算したものをそれぞれnormの格納した


#for i, row in df1.iterrows():
    #df1.loc[i,'CCALD_norm3']=np.linalg.norm(row)
    #row[i,'IPS_norm']=np.linalg.norm(df2.iloc[i])
#for文を使ったときの書き方

def Normalize1(row):#全部のベクトルをノルムで割った
    row=row/row['CCALD_norm']
    return row
df1=df1.apply(Normalize1,axis=1)

def Normalize2(row):
    row=row/row['IPS_norm']
    return row
df2=df2.apply(Normalize2,axis=1)

df1=df1.drop('CCALD_norm',axis=1)
df2=df2.drop('IPS_norm',axis=1)#normを消去

df_=pd.concat([df1,df2],axis=1)#インデックスで結合


df_.to_csv('normalized.csv')