import pandas as pd

df=pd.read_csv('data/alzheimer.csv')
df2=pd.read_csv('data/alzheimer_name.csv')
print(len(df))
df=df[~(df["IDENTIFIER"].str.contains('chr', na=False))]
#特定の文字を含む行の消去
df=df[~(df["IDENTIFIER"]=='control')]
df=df[~(df["IDENTIFIER"].str.contains('-', na=False))]

df=(df.groupby('IDENTIFIER').mean())
#重複した値の消去
print(df.head())
print(len(df))
df=pd.merge(df,df2,on='IDENTIFIER',how='inner')
df=df.drop(columns=['ID','Gene.title','ID_REF'])
df=(df.groupby('IDENTIFIER').mean())
df.to_csv('data/alzheimer_tuned.csv',index='IDENTIFIER')