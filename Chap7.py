# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 10:57:00 2017

@author: 00mymy
"""

import math
from numpy import random
from scipy.stats import norm
from scipy.stats import beta

def normal_approximation_to_binomial(n, p):
    """finds mu and sigma corresponding to a Binomial(n, p)"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma
    

def normal_probability_below(hi, mu=0, sigma=1):
    return norm.cdf(hi, loc=mu, scale=sigma)

def normal_probability_above(lo, mu=0, sigma=1):
    return 1-norm.cdf(lo, loc=mu, scale=sigma)
    

def normal_probability_between(lo, hi, mu=0, sigma=1):
    return norm.cdf(hi, loc=mu, scale=sigma)-norm.cdf(lo, loc=mu, scale=sigma)



def normal_upper_bound(probability, mu=0, sigma=1):
    """returns the z for which P(Z <= z) = probability"""
    return norm.ppf(probability, mu, sigma)
    
def normal_lower_bound(probability, mu=0, sigma=1):
    """returns the z for which P(Z >= z) = probability"""
    return norm.ppf(1 - probability, mu, sigma)
    
def normal_two_sided_bounds(probability, mu=0, sigma=1):
    """returns the symmetric (about the mean) bounds that contain the specified probability"""
    tail_probability = (1 - probability) / 2
    
    # upper bound should have tail_probability above it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)
    # lower bound should have tail_probability below it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)
    return lower_bound, upper_bound


def two_sided_p_value(x, mu=0, sigma=1):
    if x >= mu:
        # if x is greater than the mean, the tail is what's greater than x
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # if x is less than the mean, the tail is what's less than x
        return 2 * normal_probability_below(x, mu, sigma)


# A/B 테스트 - Counts --> Binomial --> Normal estimation --> more approximation
def estimated_parameters(N, n):
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma
    
def a_b_test_statistic(N_A, n_A, N_B, n_B):
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)


#Bayesian Inference
# 아래의 함수들 대신 간편하게 scipy.stats.pdf 함수를 이용해도 된다.
# from scipy.stats import beta
# beta.pdf(x, alpha, beta)
def B(alpha, beta):
    """a normalizing constant so that the total probability is 1"""
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x, alpha, beta):
    if x < 0 or x > 1: # no weight outside of [0, 1]
        return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)



'''
mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
# 95% bounds based on assumption p is 0.5
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)

# actual mu and sigma based on p = 0.55
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)
# a type 2 error means we fail to reject the null hypothesis
# which will happen when X is still in our original interval

type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probability # 0.887
#norm.pdf(0) #=normal_pdf(0)
#norm.cdf(0) #=normal_cdf(0)
#norm.ppf(.5) #=inverse_normal_cdf(.5)

# 책에 소개된 검정
hi = normal_upper_bound(0.95, mu_0, sigma_0)
# is 526 (< 531, since we need more probability in the upper tail)
type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
power = 1 - type_2_probability # 0.936

# 하지만 나는 이렇게 하고싶다
hi = normal_upper_bound(0.95, mu_0, sigma_0)
lo = normal_lower_bound(0.95, mu_0, sigma_0)
# is 526 (< 531, since we need more probability in the upper tail)
type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probability # 0.936 (거의 같음)
'''


# p-value simulation
# 교재에 나온 코드인데, 이렇게하면 10만*10만 루프를 돈다
'''
extreme_value_count = 0
for _ in range(100000):
    num_heads = sum(1 if random.random() < 0.5 else 0 # count # of heads
        for _ in range(1000)) # in 1000 flips
    if num_heads >= 530 or num_heads <= 470: # and count how often
        extreme_value_count += 1 # the # is 'extreme'
print(extreme_value_count / 100000) # 0.062
'''


# p-value simulation
# 간단하게 바꾸자. numpy.random.binomial() 이용
'''
extreme_value_count = 0
for _ in range(100000):
    num_heads = random.binomial(1000, 0.5) # in 1000 flips
    if num_heads >= 530 or num_heads <= 470: # and count how often
        extreme_value_count += 1 # the # is 'extreme'
print(extreme_value_count / 100000) # 0.062
'''

'''
# AB 테스트
z = a_b_test_statistic(1000, 500, 1000, 450) # -2.24
two_sided_p_value(z) # 0.025

z = a_b_test_statistic(1000, 500, 1000, 490) # -0.44
two_sided_p_value(z) # 0.655
'''



#Bayesian inference & beta distribution
# 실험 : 10번 던졌더니 3번 앞면
# prior : Beta(20,20)
# center=0.46
# 동전을 던졌을 때 앞면이 나올 확률이 0.36~0.56일 확률(likelihood)
p=beta.cdf(23/(23+27)+.1, 23,27)-beta.cdf(23/(23+27)-.1, 23,27) #85%


