import pandas as pd


df1=pd.read_csv('data/marfan.csv',index_col='IDENTIFIER',header=0)
df2=pd.read_csv('data/marfan_name.csv')
df3=pd.read_csv('data/marfan_label.csv')

df1=(df1.groupby('IDENTIFIER').mean())
df2=df2.drop_duplicates()
#df2は重複しているものがあるので消去した
#marfan.csvの中で重複している遺伝子を抽出して値の平均値をとった
df=pd.merge(df1,df2,on='IDENTIFIER',how='inner')

#marfan_nameで遺伝子名がはっきりしているものをmergeで抽出した
df=df.drop(columns=['ID_REF','Gene.title'])
#Gene.titleとID_REFを落とした、不要なので
df=df.rename(columns={'IDENTIFIER':'Name'})
#IDENTIFIERのカラムの名前をNameに変更

df=df.set_index("Name").transpose()
#transposeするときのindexを指定して転置する
df['IDENTIFIER']=df.index.copy()
#Nameがindexになっていて認識されないので新たにindexをコピーしたName列を作ってmergeする
df=pd.merge(df,df3,on='IDENTIFIER',how='inner')
df=df.set_index('IDENTIFIER')
#IDENTIFIERをindexに指定した
df=df.sort_values('Label')
print(df.columns,df.index)
df.to_csv('data/marfan_full.csv')