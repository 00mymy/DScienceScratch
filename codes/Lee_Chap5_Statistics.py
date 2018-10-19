# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 16:43:23 2017

@author: 00mymy
"""

import random as rd
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np

num_goals = [11,8,4,4,10,11,4,7,7,1,8,26,13,16,28,15,13,14,12,10] #1998~2017 이동국 시즌 득점수 (리그+FA컵+리그컵+대륙대회)
num_matches = [24,19,8,26,26,27,25,28,10,24,17,36,40,38,48,40,40,41,38,31]

num_values = num_matches
num_counts = Counter(num_values)

xs = range(max(num_values) + 1) # largest value
ys = [num_counts[x] for x in xs] # height is just # of friends
plt.bar(xs, ys)
plt.axis([0, max(num_values)+1, 0, max(num_counts.values())+1])
plt.title("Histogram of Values")
plt.xlabel("Year")
plt.ylabel("# of Goals")
plt.show()


'''
plt.boxplot([num_goals],  0, 'bD', showmeans=True)
'''



num_points = len(num_values)
largest_value = max(num_values)
smallest_value = min(num_values)

sorted_values = sorted(num_values)
smallest_value = sorted_values[0]
second_smallest_value = sorted_values[1]
second_largest_value = sorted_values[-2]



#np.mean(x)
def mean(x):
    return sum(x) / len(x)


#np.median(x)
def median(v):
    """finds the 'middle-most' value of v"""
    n = len(v)
    sorted_v = sorted(v)
    midpoint = n // 2
    if n % 2 == 1:
        # if odd, return the middle value
        return sorted_v[midpoint]
    else:
        # if even, return the average of the middle values
        lo = midpoint - 1
        hi = midpoint
        return (sorted_v[lo] + sorted_v[hi]) / 2

#np.percentile() 
def quantile(x, p):
    """returns the pth-percentile value in x"""
    p_index = int(p * len(x))
    return sorted(x)[p_index]


def mode(x):
    """returns a list, might be more than one mode"""
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()
                    if count == max_count]

'''
def mode(x):
    """returns a list, might be more than one mode"""
    counts = Counter(x)
    return counts.most_common(1)[0]
'''

quantile(num_values, 0.99)



#
# Dispersion
#
def data_range(x):
    return max(x) - min(x)

def de_mean(x):
    """translate x by subtracting its mean (so the result has mean 0)"""
    x_bar = sum(x)/len(x) #mean
    return [x_i - x_bar for x_i in x]

#np.var(x)    
def variance(x):
    """assumes x has at least two elements"""
    n = len(x)
    deviations = de_mean(x)
    return sum(d*d for d in deviations) / (n-1)

   
import math
#np.std(x)
def standard_deviation(x):
    return math.sqrt(variance(x))

def interquartile_range(x):
    return quantile(x, 0.75) - quantile(x, 0.25)


'''
plt.boxplot([num_goals],  0, 'bD', showmeans=True)
'''


def covariance(x, y):
    n = len(x)
    return np.dot(de_mean(x), de_mean(y)) / (n-1)

covariance([1,1,1], [2,2,2])
covariance([1,2,3], [10,20,30])
covariance([1,2,3], [100,200,300])
covariance([1,2,3], [3,4,5])
covariance([1,2,3], [5,4,3])



def correlation(x, y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0 # if no variation, correlation is zero
        
correlation([1,1,1], [2,2,2])
correlation([1,2,3], [10,20,30])
correlation([1,2,3], [100,200,300])
correlation([1,2,3], [3,4,5])
correlation([1,2,3], [5,4,3])