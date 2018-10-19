# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:24:39 2017

@author: 00mymy

Ref)
Predicting customer churn with scikit-learn
http://blog.yhat.com/posts/predicting-customer-churn-with-sklearn.html
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# 데이터 획득 / 확인
SRC_FILE = 'F:/Work/workspace/Datasets/telco_churn.txt'
churn_df = pd.read_csv(SRC_FILE)
col_names = churn_df.columns.tolist()

print("Column names:")
print(col_names)

to_show = col_names[:6] + col_names[-6:]

print("\nSample data:")
churn_df[to_show].head(6)


# 데이터 변환 및 정제 - 모델에서 다루기 쉽게, 모델에 적용 가능하게

# 'True.'/'False.'  --> 1/0
churn_result = churn_df['Churn?']
y = np.where(churn_result == 'True.',1,0)

# 필요없는 필드 제거
to_drop = ['State','Area Code','Phone','Churn?']
churn_feat_space = churn_df.drop(to_drop,axis=1) # axix=1 means 'column'

# 'yes'/'no' has to be converted to boolean values
# NumPy converts these from boolean to 1. and 0. later
yes_no_cols = ["Int'l Plan","VMail Plan"]
churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'

# Pull out features for future use
features = churn_feat_space.columns

X = churn_feat_space.as_matrix().astype(np.float)
# NumPy converts boolean to 1. and 0.


# This is important
# Scaling (normalizing each feature to a range of around 1.0 to -1.0)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Feature space holds %d observations and %d features" % X.shape)
print("Unique target labels:", np.unique(y))

# 여기까지 데이터 획득 및 전처리 완료
#  feature space 'X' and a set of target values 'y'


# http://scikit-learn.org/stable/modules/cross_validation.html
# 일반적으로 Training set, Test set 또는 Training set, Valiation Set, Test set으로 나누는 대신
# KFold Cross-validation 방식을 사용한다.
# 보통은 다음과 같이 train_test_split 기능을 사용한다.
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold

def run_cv(X,y,clf_class,**kwargs):
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=5,shuffle=True)
    y_pred = y.copy()

    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred
    

# 예측 모델들 여러개를 가지고 실험해 본다.
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN

def accuracy(y_true,y_pred):
    # NumPy interprets True and False as 1. and 0.
    return np.mean(y_true == y_pred)

# Accuracy 측정
print("Support vector machines:")
print("%.3f" % accuracy(y, run_cv(X,y,SVC)))
print("Random forest:")
print("%.3f" % accuracy(y, run_cv(X,y,RF)))
print("K-nearest-neighbors:")
print("%.3f" % accuracy(y, run_cv(X,y,KNN)))


# Confusion Matrix
from sklearn.metrics import confusion_matrix

y = np.array(y)
class_names = np.unique(y)

confusion_matrices = [
    ( "Support Vector Machines", confusion_matrix(y,run_cv(X,y,SVC)) ),
    ( "Random Forest", confusion_matrix(y,run_cv(X,y,RF)) ),
    ( "K-Nearest-Neighbors", confusion_matrix(y,run_cv(X,y,KNN)) ),
]


from plot_confusion_matrix import getConfusionMatrixPlot

plot_svc = getConfusionMatrixPlot(y, run_cv(X,y,SVC), class_names)
plot_svc.title('SVC')
plot_svc.show()

plot_rf = getConfusionMatrixPlot(y, run_cv(X,y,RF), class_names)
plot_rf.title('RF')
plot_rf.show()

plot_knn = getConfusionMatrixPlot(y, run_cv(X,y,KNN), class_names)
plot_knn.title('kNN')
plot_knn.show()



def run_prob_cv(X, y, clf_class, **kwargs):
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_prob = np.zeros((len(y),2))
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
    return y_prob


import warnings
warnings.filterwarnings('ignore')

# Use 10 estimators so predictions are all multiples of 0.1
pred_prob = run_prob_cv(X, y, RF, n_estimators=10)
pred_churn = pred_prob[:,1]
is_churn = y == 1

# Number of times a predicted probability is assigned to an observation
counts = pd.value_counts(pred_churn)

# calculate true probabilities
true_prob = {}
for prob in counts.index:
    true_prob[prob] = np.mean(is_churn[pred_churn == prob])
    true_prob = pd.Series(true_prob)

# pandas-fu
counts = pd.concat([counts,true_prob], axis=1).reset_index()
counts.columns = ['pred_prob', 'count', 'true_prob']
counts

'''
#하여간... ggplot 졸라 안맞아...
from ggplot import *
#%matplotlib inline

baseline = np.mean(is_churn)
ggplot(counts,aes(x='pred_prob',y='true_prob',size='count')) + \
    geom_point(color='blue') + \
    stat_function(fun = lambda x: x, color='red') + \
    stat_function(fun = lambda x: baseline, color='green') + \
    xlim(-0.05,  1.05) + ylim(-0.05,1.05) + \
    ggtitle("Random Forest") + \
    xlab("Predicted probability") + ylab("Relative frequency of outcome")
'''

baseline = np.mean(is_churn)
px = counts['pred_prob']
py = counts['true_prob']
ps = counts['count'] # for dot sizing
plt.axis([-0.1,1.1,-0.1,1.1])
plt.scatter(px,py, s=ps/2)
plt.plot(counts.index/10, counts.index/10, color='r')
plt.plot(counts.index/10, [baseline]*len(counts.index), color='g')