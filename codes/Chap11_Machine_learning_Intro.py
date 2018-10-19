# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 20:13:25 2017

@author: 00mymy
"""
import random
import numpy as np

def split_data(data, prob):
    """split data into fractions [prob], [1 - prob]"""
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results
    
def train_test_split(X, y, test_pct):
    ''' X : input matrix, y : output vector '''
    data = zip(X, y) # pair corresponding values
    train, test = split_data(data, 1 - test_pct) # split the data set of pairs
    X_train, y_train = zip(*train) # magical un-zip trick
    X_test, y_test = zip(*test)
    return X_train, X_test, y_train, y_test

#Test Example    
X = []
y = []
for i in range(100):
    X.append([int(np.random.normal(loc=55, scale=15)) for j in range(10)])
    y.append(1 if sum(X[i]) > 600 else 0)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_pct=.3)



#
# correctness
#

def accuracy(tp, fp, fn, tn):
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total

def precision(tp, fp, fn, tn):
    return tp / (tp + fp)

def recall(tp, fp, fn, tn):
    return tp / (tp + fn)

def f1_score(tp, fp, fn, tn):
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)

    return 2 * p * r / (p + r)


'''
            leukemia    no leukemia     total
“Luke”      70          4,930           5,000
not “Luke”  13,930      981,070         995,000
total       14,000      986,000         1,000,000
'''
print("accuracy(70, 4930, 13930, 981070)", accuracy(70, 4930, 13930, 981070))
print("precision(70, 4930, 13930, 981070)", precision(70, 4930, 13930, 981070))
print("recall(70, 4930, 13930, 981070)", recall(70, 4930, 13930, 981070))
print("f1_score(70, 4930, 13930, 981070)", f1_score(70, 4930, 13930, 981070))
