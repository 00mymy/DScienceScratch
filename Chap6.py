# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 10:31:00 2017

@author: 00mymy
"""

from numpy import random as nrd

def random_suldam():
    return nrd.choice(["sul", "dam", "both", "none"], p=[.8, .1, .09, .01])
    
both_cnt = 0
sul_cnt = 0
dam_cnt = 0
none_cnt = 0
nrd.seed(0)

sample_size = 10000
for _ in range(sample_size):
    suldam = random_suldam()
    
    if suldam == "both":
        both_cnt += 1
    elif suldam == "sul":
        sul_cnt += 1
    elif suldam == "dam":
        dam_cnt += 1
    elif suldam == "none":
        none_cnt += 1

print("P(dam|sul):", both_cnt / sul_cnt) 
print("P(sul|dam): ", both_cnt / dam_cnt)



from collections import Counter
from matplotlib import pyplot as plt

bn_counts = Counter([nrd.binomial(100, .5) for _ in range(10000)])

xs = range(0, 100, 1)
ys = [bn_counts[x]/10000 for x in xs]
plt.plot(xs,ys)
plt.title("Binomial Distribution vs. Normal Approximation")
plt.show()



# 책에 소개하는 방법
import math

def uniform_pdf(x):
    return 1 if x >= 0 and x < 1 else 0

def uniform_cdf(x):
    "returns the probability that a uniform random variable is <= x"
    if x < 0: return 0 # uniform random is never less than 0
    elif x < 1: return x # e.g. P(X <= 0.4) = 0.4
    else: return 1 # uniform random is always less than 1

def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma))

def normal_cdf(x, mu=0,sigma=1):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2
    
def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    """find approximate inverse using binary search"""
    # if not standard, compute standard and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
        
    low_z, low_p = -10.0, 0 # normal_cdf(-10) is (very close to) 0
    hi_z, hi_p = 10.0, 1 # normal_cdf(10) is (very close to) 1
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2 # consider the midpoint
        mid_p = normal_cdf(mid_z) # and the cdf's value there
        if mid_p < p:
            # midpoint is still too low, search above it
            low_z, low_p = mid_z, mid_p
        elif mid_p > p:
            # midpoint is still too high, search below it
            hi_z, hi_p = mid_z, mid_p
        else:
            break
    return mid_z


# scipy.stats 라이브러리를 이용하면 간편하다
from scipy.stats import norm
norm.pdf(0) #=normal_pdf(0)
norm.cdf(0) #=normal_cdf(0)
norm.ppf(.5) #=inverse_normal_cdf(.5)