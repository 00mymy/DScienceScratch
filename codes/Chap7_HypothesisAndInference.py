# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 10:57:00 2017

@author: 00mymy
"""

import math
from numpy import random
from scipy.stats import norm
from scipy.stats import beta

# binomial --> normal
def normal_approximation_to_binomial(, p):
    """finds mu and sigma correspondinng to a Binomial(n, p)"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma
    

# 변수값 -> 확률 구하기
def normal_probability_below(hi, mu=0, sigma=1):
    return norm.cdf(hi, loc=mu, scale=sigma)

def normal_probability_above(lo, mu=0, sigma=1):
    return 1-norm.cdf(lo, loc=mu, scale=sigma)
    

def normal_probability_between(lo, hi, mu=0, sigma=1):
    return norm.cdf(hi, loc=mu, scale=sigma)-norm.cdf(lo, loc=mu, scale=sigma)


# 확률 -> 변수값 구하기 
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


'''
# binomial --> normal
#(500.0, 15.811388300841896)
mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)

# 95% bounds based on assumption p is 0.5
#(469.0102483847719, 530.9897516152281)
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)


# actual mu and sigma based on p = 0.55
# (550.0, 15.732132722552274)
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)

# a type 2 error means we fail to reject the null hypothesis
# which will happen when X is still in our original interval
type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1) #0.113
power = 1 - type_2_probability # 0.887
'''


'''
# 단측검정 (책에 소개된 검정)
hi = normal_upper_bound(0.95, mu_0, sigma_0)
# is 526 (< 531, since we need more probability in the upper tail)
type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
power = 1 - type_2_probability # 0.936

# 양측검정할 경우 (차이점 비교-해석하기 나름)
hi = normal_upper_bound(0.975, mu_0, sigma_0)
lo = normal_lower_bound(0.975, mu_0, sigma_0)
# is 526 (< 531, since we need more probability in the upper tail)
type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probability # 0.887
'''



#p-value
def two_sided_p_value(x, mu=0, sigma=1):
    if x >= mu:
        # if x is greater than the mean, the tail is what's greater than x
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # if x is less than the mean, the tail is what's less than x
        return 2 * normal_probability_below(x, mu, sigma)


'''
two_sided_p_value(531.5, mu_0, sigma_0) 	# 0.0463

upper_p_value = normal_probability_above
lower_p_value = normal_probability_below
upper_p_value(524.5, mu_0, sigma_0) 		# 0.061
upper_p_value(526.5, mu_0, sigma_0) 		# 0.047

'''

# p-value simulation
# 교재에 나온 코드
'''
extreme_value_count = 0
for _ in range(100000):
    num_heads = sum(1 if random.random() < 0.5 else 0 # count # of heads
        for _ in range(1000)) # in 1000 flips
    if num_heads >= 531 or num_heads <= 468: # and count how often
        extreme_value_count += 1 # the # is 'extreme'
print(extreme_value_count / 100000) # 0.062 (> 0.05, don't reject)
'''


# p-value simulation
# 간단하게 바꾸자. numpy.random.binomial() 이용
# extream 값 범위를 살짝 높여(낮춰)보자
'''
extreme_value_count = 0
for _ in range(100000):
    num_heads = random.binomial(1000, 0.5) # in 1000 flips
    if num_heads >= 532 or num_heads <= 468: # and count how often
        extreme_value_count += 1 # the # is 'extreme'
print(extreme_value_count / 100000) # 0.062 
'''

'''
# Confidence Interval
p_hat = 525 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000) # 0.0158
normal_two_sided_bounds(0.95, mu, sigma) # [0.4940, 0.5560], 0.5 in the interval (fair range)

p_hat = 540 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000) # 0.0158
normal_two_sided_bounds(0.95, mu, sigma) # [0.5091, 0.5709] (unfair range)
'''

# A/B 테스트 - Counts --> Binomial --> Normal estimation --> more approximation
def estimated_parameters(N, n):
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma
    
def a_b_test_statistic(N_A, n_A, N_B, n_B):
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)


'''
# AB 테스트
z = a_b_test_statistic(1000, 500, 1000, 450) # -2.24
two_sided_p_value(z) # 0.025

z = a_b_test_statistic(1000, 500, 1000, 490) # -0.44
two_sided_p_value(z) # 0.655
'''


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


#Bayesian inference & beta distribution
# 실험 : 10번 던졌더니 3번 앞면
# prior : Beta(20,20)
# center=0.46
# 동전을 던졌을 때 앞면이 나올 확률이 0.36~0.56일 확률(likelihood)
#from scipy.stats import beta
p=beta.cdf(23/(23+27)+.1, 23,27)-beta.cdf(23/(23+27)-.1, 23,27) #84.6%


