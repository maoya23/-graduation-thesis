import pandas as pd

df1=pd.read_csv('data/Alzheimer.T.csv')
#Brunner_test.pyでは抽出した後にラベルが消えるので、手動でラベル追加
df2=pd.read_csv('data/expand.csv')

df1=df1['IDENTIFIER']
df2=df2['IDENTIFIER']
df2 = df2.str.strip()
#これで空白を埋めた
df = pd.merge(df1, df2,on='IDENTIFIER' ,how='inner')
print(len(df))

print(df.head())
df.to_csv('data/alzheimer_tunedGene.csv')