# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 16:43:23 2017

@author: 00mymy
"""

import random as rd
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np

'''
num_friends = [100, 49, 41, 40, 25,
               34, 12, 4, 5, 13,
               11, 14, 34, 18, 22,
               87, 59, 26, 3, 1,
               91, 63, 28, 55, 45,
               32, 61, 49, 34, 17,
               48, 18, 36, 35, 66,
               ]
'''
num_friends = [int(rd.gauss(50,10)) for _ in range (200)]
#num_friends = [int(np.random.lognormal(3,.5)) for _ in range (200)]
np.random.lognormal()
'''
friend_counts = Counter(num_friends)

xs = range(101) # largest value is 100
ys = [friend_counts[x] for x in xs] # height is just # of friends
plt.bar(xs, ys)
plt.axis([0, 101, 0, 25])
plt.title("Histogram of Friend Counts")
plt.xlabel("# of friends")
plt.ylabel("# of people")
plt.show()
'''

'''
plt.boxplot([num_friends],  0, 'bD', showmeans=True)
'''

'''
num_points = len(num_friends)
largest_value = max(num_friends)
smallest_value = min(num_friends)

sorted_values = sorted(num_friends)
smallest_value = sorted_values[0]
second_smallest_value = sorted_values[1]
second_largest_value = sorted_values[-2]
'''


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
plt.boxplot([num_friends],  0, 'bD', showmeans=True)
'''

    
def covariance(x, y):
    n = len(x)
    return np.dot(de_mean(x), de_mean(y)) / (n-1)

def correlation(x, y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0 # if no variation, correlation is zero