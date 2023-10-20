import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (roc_curve, auc, accuracy_score)

df=pd.read_csv('data/marfan.T.csv',index_col='IDENTIFIER')

train_X=df.drop('Label',axis=1)
train_y=df.Label

(train_X,test_X,train_y,test_y)=train_test_split(train_X,train_y,test_size=0.3,random_state=666)

clf=DecisionTreeClassifier(random_state=0)
clf=clf.fit(train_X,train_y)

pred = clf.predict(test_X)
fpr, tpr, thresholds = roc_curve(test_y, pred, pos_label=1)
auc(fpr, tpr)
print(accuracy_score(pred, test_y))

# ROC曲線のプロット
#plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr, tpr, label='Logistic Regression')
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve')
#plt.savefig('graph/marfan_ROC.png')


features=train_X.columns
importances=clf.feature_importances_
importances=importances[importances>0.1]#得られた結果はnaddary配列なのでpd.queryは使えないのでブールインデックスで抽出
indices=np.argsort(importances)

plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.savefig('graph/marfan_randomforest.png')