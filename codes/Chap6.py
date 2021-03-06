# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 10:31:00 2017

@author: 00mymy
"""

from numpy import random as nrd
import math
import random

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


# uniform distribution - pdf
#plt.plot([-1,0,0,1, 1,2], [0,0,1,1,0,0])


# uniform distribution - cdf
#plt.plot([-1,0,0,1, 1,2], [0,0,0,1,1,1])


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


'''
xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_pdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend()
plt.title("Various Normal pdfs")
plt.show()
'''

# scipy.stats 라이브러리를 이용하면 간편하다
'''
from scipy.stats import norm
norm.pdf(0) #=normal_pdf(0)
norm.cdf(0) #=normal_cdf(0)
norm.ppf(.5) #=inverse_normal_cdf(.5)

norm.cdf(1)-norm.cdf(-1)
norm.cdf(2)-norm.cdf(-2)
norm.cdf(3)-norm.cdf(-3)
norm.cdf(6)-norm.cdf(-6) # 6-sigma

# 1 sigma partition : 16 + 68 + 16
norm.ppf(.84)
norm.ppf([.16, .84]) #68% 구간. 비교: norm.cdf(1)-norm.cdf(-1)

norm.ppf(.95, loc=10,scale=.1)
norm.cdf(10.1645, loc=10,scale=.1) - norm.cdf(9.8355, loc=10,scale=.1) #90% 구간 (양쪽으로 +- 5% 컷)
norm.ppf([.025,.975], loc=10,scale=.1) #95% 구간





# partial
from functools import partial
c10 = partial(norm.cdf, loc=10,scale=.1)
p10 = partial(norm.ppf, loc=10,scale=.1)
p10(.95)
c10(10.1645)-c10(9.8355) 


xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs,[norm.pdf(x,loc=0, scale=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[norm.pdf(x,loc=0, scale=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[norm.pdf(x,loc=0, scale=.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[norm.pdf(x,loc=-1, scale=1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend()
plt.title("Various Normal pdfs")
plt.show()


plt.plot(xs,[norm.cdf(x,loc=0, scale=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[norm.cdf(x,loc=0, scale=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[norm.cdf(x,loc=0, scale=.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[norm.cdf(x,loc=-1, scale=1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend()
plt.title("Various Normal pdfs")
plt.show()
'''

# Central Limit Theorem
def bernoulli_trial(p):
    return 1 if random.random() < p else 0
    
def binomial(n, p):
    return sum(bernoulli_trial(p) for _ in range(n))

def make_hist_binomial(p, n, num_points):
   data = [binomial(n, p) for _ in range(num_points)]
   
   # use a bar chart to show the actual binomial samples
   histogram = Counter(data)
   plt.bar([x - 0.4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()], # 확률(빈도)
            0.8,
            color='0.75')
   
   mu = p * n
   sigma = math.sqrt(n * p * (1 - p))
   # use a line chart to show the normal approximation
   xs = range(min(data), max(data) + 1)
   ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma) for i in xs]
   plt.plot(xs,ys)
   plt.title("Binomial Distribution vs. Normal Approximation")
   plt.show()


make_hist_binomial(0.75, 100, 10000)


'''
bn_counts = Counter([nrd.binomial(100, .75) for _ in range(10000)])

xs = range(20,120)
ys = [bn_counts[x] for x in xs]
plt.plot(xs,ys)
plt.title("Binomial Distribution vs. Normal Approximation")
plt.show()
'''

# Random 비교
'''
#nums = [int(nrd.rand()*100) for _ in range(1000)]
#nums = [int(nrd.uniform(0,101)) for _ in range(1000)]
#nums = [int(nrd.normal(loc=50, scale=10)) for _ in range(1000)]
#nums = [int(nrd.binomial(100,.75)) for _ in range(1000)]
nums = [int(nrd.lognormal(mean=4, sigma=.2)) for n in range(1000)]

num_counts = Counter(nums)
xs = range(101) # largest value is 100
ys = [num_counts[x] for x in xs] # height is just # of friends
plt.bar(xs, ys)
plt.axis([0, 101, 0, 150])
plt.title("Histogram of Random Number Counts")
plt.xlabel("Random numbers")
plt.ylabel("# of times")
plt.show()
'''

'''
# Binomial vs. Normal
nums_b = [int(nrd.binomial(100,.75)) for _ in range(1000)]
nums_n = [int(nrd.normal(loc=75, scale=4.33)) for _ in range(1000)]
counts_b = Counter(nums_b)
counts_n = Counter(nums_n)

xs = range(20,120)
plt.plot(xs,[counts_b[x] for x in xs],':')
plt.plot(xs,[counts_n[x] for x in xs],'-.')
plt.show()

'''
