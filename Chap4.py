# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 17:27:37 2017

@author: 00mymy
"""

def vector_add(v, w):
    """adds corresponding elements"""
    return [v_i + w_i for v_i, w_i in zip(v, w)]
    

def vector_subtract(v, w):
    """subtracts corresponding elements"""
    return [v_i - w_i
        for v_i, w_i in zip(v, w)]
            
def vector_sum(vectors):
    """sums all corresponding elements"""
    result = vectors[0] # start with the first vector
    for vector in vectors[1:]: # then loop over the others
        result = vector_add(result, vector) # and add them to the result
    return result
            
def scalar_multiply(c, v):
    """c is a number, v is a vector"""
    return [c * v_i for v_i in v]
    
def vector_mean(vectors):
    """compute the vector whose ith element is the mean of theith elements of the input vectors"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

def dot(v, w):
    """v_1 * w_1 + ... + v_n * w_n"""
    return sum(v_i * w_i
            for v_i, w_i in zip(v, w))
                
def sum_of_squares(v):
    """v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)
    
import math
def magnitude(v):
    return math.sqrt(sum_of_squares(v)) # math.sqrt is square root function

def squared_distance(v, w):
    """(v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(vector_subtract(v, w))
'''    
def distance(v, w):
    return math.sqrt(squared_distance(v, w))
'''
    
def distance(v, w):
    return magnitude(vector_subtract(v, w))


# numpy array - 편리하게...
# http://www.scipy-lectures.org/intro/numpy/operations.html
'''
import numpy as np
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

v1+v2
v1-v2
v1+1
3*v1
v1*v2
sum(v1)
v1.sum() 
v2.max()
v2.argmax()
v1.argmin()
v2.mean()
np.mean(v2)
np.median(v2)
v2.std()
np.std(v2)
v1.dot(v2)
np.dot(v1,v2) # 'dot' operation is not array multiplication

# scipy
from scipy.spatial import distance as dist
d1 = dist.euclidean(v1,v2)
d2 = dist.cosine(v1,v2)
d3 = dist.correlation(v1,v2)
'''




A = [[1, 2, 3], # A has 2 rows and 3 columns
     [4, 5, 6]]
     
B = [[1, 2], # B has 3 rows and 2 columns
     [3, 4],
     [5, 6]]
     
import functools as ft
sum = ft.reduce(lambda a, b : 3, [1,2,3,4,5])



def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) #if A else 0 # number of elements in first row
    return num_rows, num_cols
    
def get_row(A, i):
    return A[i] # A[i] is already the ith row

def get_column(A, j):
    return [A_i[j] # jth element of row A_i
        for A_i in A] # for each row A_i
        
        
def make_matrix(num_rows, num_cols, entry_fn):
    return [[entry_fn(i, j) # given i, create a list
        for j in range(num_cols)] # [entry_fn(i, 0), ... ]
            for i in range(num_rows)] # create one list for each i
            
def is_diagonal(i, j):
    """1's on the 'diagonal', 0's everywhere else"""
    return 1 if i == j else 0

identity_matrix = make_matrix(5, 5, is_diagonal)

#heights, weights, and ages of 1,000 people you could put them in a 1, 000 × 3 matrix
data = [[70, 170, 40],
[65, 120, 26],
[77, 250, 19],
# ....
]




friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
(4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]


#We could also represent this as:
# user 0 1 2 3 4 5 6 7 8 9
#
friendships = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0], # user 0
[1, 0, 1, 1, 0, 0, 0, 0, 0, 0], # user 1
[1, 1, 0, 1, 0, 0, 0, 0, 0, 0], # user 2
[0, 1, 1, 0, 1, 0, 0, 0, 0, 0], # user 3
[0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # user 4
[0, 0, 0, 0, 1, 0, 1, 1, 0, 0], # user 5
[0, 0, 0, 0, 0, 1, 0, 0, 1, 0], # user 6
[0, 0, 0, 0, 0, 1, 0, 0, 1, 0], # user 7
[0, 0, 0, 0, 0, 0, 1, 1, 0, 1], # user 8
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]] # user 9

friends_of_five = [i for i, is_friend in enumerate(friendships[5])
    if is_friend]

'''
def friends_of(p) :
    return [i for i, is_friend in enumerate(friendships[p])
                if is_friend]
'''