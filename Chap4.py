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
