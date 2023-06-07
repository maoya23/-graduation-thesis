import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('data/Alzheimer.T.csv',index_col='IDENTIFIER')
df=df.sort_values('p_value',ascending=True)#分散の値が高いもの順に並び替え
#1から13621の文字列をnumpy配列で生成

fig = plt.figure()
ax=fig.add_subplot(1,1,1)
ax.set_xlabel('p_value')
ax.set_ylabel('number of genes')
y=df['p_value']
ax.hist(y,bins=100, log=True,ec='black')
#ax.scatter(x,y)
plt.gca().spines['right'].set_visible(False)#グラフの上側の線を消す
plt.gca().spines['top'].set_visible(False)


#plt.grid(True)
plt.savefig('graph/alzheimer_brunner.png')