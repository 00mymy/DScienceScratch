# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 11:22:53 2017

@author: 00mymy
"""

for i in [1, 2, 3, 4, 5]:
    print('i : %d' % i)                     # first line in "for i" block
    for j in [1, 2, 3, 4, 5]:
        print('-- j : %d' % j)              # first line in "for j" block
        print('-- i+j :%d' % (i + j))       # last line in "for j" block
    print('last i : %d' % i)               # last line in "for i" block
print("END - done looping")

long_winded_computation = (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 +
13 + 14 + 15 + 16 + 17 + 18 + 19 + 20)

list_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

easier_to_read_list_of_lists = [[1, 2, 3],
                                [4, 5, 6],
                    [7, 8, 9] ]

# Error
two_plus_three = 2 + 
3

# No error - line continues
two_plus_three = 2 + \
3

import re
my_regex = re.compile("[0-9]+", re.I) 

import re as regex
my_regex = regex.compile("[0-9]+", regex.I)

my_regex.findall('aaaaa123bbbbb456')


import matplotlib.pyplot as plt
plt.plot([1,2,3,4,5], [10,20,30,40,50])
plt.plot([1,2,3,4,5], [i**2 for i in [1,2,3,4,5]])


from collections import defaultdict, Counter
lookup = defaultdict(int)
my_counter = Counter()


match = 10
from re import * # uh oh, re has a match function
print(match) # "<function re.match>"



#####################################################################
# Functions

def double(x):
    return x * 2
    
def square(x):
    return x * x
    
def apply_to_one(f):
    """calls the function f with 1 as its argument"""
    return f(1)
    
my_double = double # refers to the previously defined function
x = apply_to_one(my_double) # equals 2

y = apply_to_one(square)

another_double = lambda x: 2 * x # don't do this
#def another_double(x): return 2 * x # do this instead

# lambda 함수는 이럴 때 (1회용)
z = apply_to_one(lambda x: 3*x)


def my_print(message="my default message"):
    print(message)

def subtract(a=0, b=0):
    return a - b
    
subtract(10, 5) # returns 5
subtract(0, 5) # returns -5
subtract(b=5) # same as previous (a=0 as default)




try:
    print(0 /0)
except ZeroDivisionError:
    print("cannot divide by zero")
    
    
    
#####################################################################
# Exceptions

#unhandled (Error)
print(0 / 0)

# handled
try:
    print(0 / 0)
except ZeroDivisionError:
    print("cannot divide by zero")


    
#####################################################################
# Lists

integer_list = [1, 2, 3]
heterogeneous_list = ["string", 0.1, True]
list_of_lists = [ integer_list, heterogeneous_list, [] ]
list_length = len(integer_list) # equals 3
list_sum = sum(integer_list) # equals 6


# x = range(10)
x = [i for i in range(10)]
zero = x[0]
one = x[1]
nine = x[-1] # 'Pythonic' for last element
eight = x[-2]
x[0] = -1

first_three = x[:3]
three_to_end = x[3:]
one_to_four = x[1:5]
last_three = x[-3:]
without_first_and_last = x[1:-1]
copy_of_x = x[:]




1 in [1, 2, 3] # True
0 in [1, 2, 3] # False

x = [1, 2, 3]
x.extend([4, 5, 6]) # x is now [1,2,3,4,5,6]

x = [1, 2, 3]
y = x + [4, 5, 6] # y is [1, 2, 3, 4, 5, 6]; x is unchanged

x = [1, 2, 3]
x.append(0) # x is now [1, 2, 3, 0]


#####################################################################
# Tuples

my_list = [1, 2]
my_tuple = (1, 2)
other_tuple = 3, 4
my_list[1] = 3 # my_list is now [1, 3]

try:
    my_tuple[1] = 3
except TypeError:
    print("cannot modify a tuple")
    
def sum_and_product(x, y):
    return (x + y),(x * y)
    
sp = sum_and_product(2, 3) # equals (5, 6)
s, p = sum_and_product(5, 10) # s is 15, p is 50

x, y = 1, 2
x, y = y, x


#####################################################################
# Dictionary

empty_dict = {} # Pythonic
empty_dict2 = dict() # less Pythonic
grades = { "Joel" : 80, "Tim" : 95 }
joels_grade = grades["Joel"]

try:
    kates_grade = grades["Kate"]
except KeyError:
    print("no grade for Kate!")

# assignment
grades['Kate'] = 90    

'Joel' in grades
'Jola' in grades

   
tweet = {
    "user" : "joelgrus",
    "text" : "Data Science is Awesome",
    "retweet_count" : 100,
    "hashtags" : ["#data", "#science", "#datascience", "#awesome", "#yolo"]
}

tweet_keys = tweet.keys() # list of keys
tweet_values = tweet.values() # list of values
tweet_items = tweet.items() # list of (key, value) tuples
"user" in tweet_keys # True, but uses a slow list in
"user" in tweet # more Pythonic, uses faster dict in
"joelgrus" in tweet_values # True
('user', 'joelgrus') in tweet_items


document = ["data", "science", "datascience", "awesome", "awesome", "yolo", "yolo", "yolo"]

word_counts = {}
for word in document:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1

'''
word_counts = {}
for word in document:
    try:
        word_counts[word] += 1
    except KeyError:
        word_counts[word] = 1
'''
'''
word_counts = {}
for word in document:
    previous_count = word_counts.get(word, 0)   # default value 0 (중요함)
    word_counts[word] = previous_count + 1
'''

from collections import defaultdict
word_counts = defaultdict(int)  # int() produces 0

# test it
word_counts['xxx']

# clean and simple!
for word in document:
	word_counts[word] += 1
 
dd_list = defaultdict(list) # list() produces an empty list
dd_list[2].append(1) # now dd_list contains {2: [1]}

dd_dict = defaultdict(dict) # dict() produces an empty dict
dd_dict["Joel"]["City"] = "Seattle" # { "Joel" : { "City" : Seattle"}}

dd_pair = defaultdict(lambda: [0, 0])
dd_pair[2][1] = 1 # now dd_pair contains {2: [0,1]}
dd_pair[2]
dd_pair[3]


#####################################################################
# Counter

from collections import Counter
c = Counter([0, 1, 2, 0])
c[0]


word_counts = Counter(document)

for word, count in word_counts.most_common(2):
    print(word, count)
    
'''
for wc in word_counts:
    print(wc, word_counts[wc])
'''


#####################################################################
# Sets
s = set()
s.add(1) # s is now { 1 }
s.add(2) # s is now { 1, 2 }
s.add(2) # s is still { 1, 2 }
x = len(s) # equals 2
y = 2 in s # equals True
z = 3 in s # equals False

hundreds_of_other_words = ['hundreds_of_other_words']
stopwords_list = ["a","an","at"] + hundreds_of_other_words + ["yet", "you"]

# faster look-up (set vs. list)
"zip" in stopwords_list # False, but have to check every element
stopwords_set = set(stopwords_list)
"zip" in stopwords_set # very fast to check

#distinct
item_list = [1, 2, 3, 1, 2, 3]
num_items = len(item_list) # 6
item_set = set(item_list) # {1, 2, 3}
num_distinct_items = len(item_set) # 3
distinct_item_list = list(item_set) # [1, 2, 3]



#####################################################################
# Control flow
if 1 > 2:
    message = "if only 1 were greater than two..."
elif 1 > 3:
    message = "elif stands for 'else if'"
else:
    message = "when all else fails use else (if you want to)"
    
parity = "even" if x % 2 == 0 else "odd"


x = 0
while x < 10:
    print("%d is less than 10" % x)
    x += 1
    
for x in range(10):
    print(x, "is less than 10")

    
for x in range(10):
    if x == 3:
        continue 
    if x == 5:
        break 
    print(x)



    
#####################################################################
# True / False

one_is_less_than_two = 1 < 2 # equals True
true_equals_false = True == False # equals False

x = None
print(x == None)
print(x is not None)

s= 'a' 
#if len(s): 
#if len(s) > 0:
if s:
    first_char = s[0]
else:
    first_char = ""
    
# More falsy examples
    '''
if None:
    print('xxx')

if not None:
    print('xxx')

#some_list = [1,2, 3]
some_list = []
if some_list:   #if len(some_list): 또는 if len(some_list) > 0
    print("The list has", len(some_list), "items")
else:
    print("The list is empty")
'''
    

#####################################################################
# Sorting
x= [4,1,2,3]
y = sorted(x) # is [1,2,3,4], x is unchanged

x.sort() #changed


# sort the list by absolute value from largest to smallest
x = sorted([-4,1,-2,3], key=abs, reverse=True) # is [-4,3,-2,1]

# sort the words and counts from highest count to lowest
wc = sorted(word_counts.items(),
            key=lambda wc_item: wc_item[1],
            reverse=True)
            

#####################################################################
# List comprehension          
even_numbers = [x for x in range(5) if x % 2 == 0] # [0, 2, 4]
squares = [x * x for x in range(5)] # [0, 1, 4, 9, 16]
even_squares = [x * x for x in even_numbers] # [0, 4, 16]

square_dict = { x : x * x for x in range(5) } # { 0:0, 1:1, 2:4, 3:9, 4:16 }
square_set = { x * x for x in [1, -1] } # { 1 }


zeroes = [0 for _ in even_numbers]
pairs = [(x, y)
    for x in range(10)
        for y in range(10)]
            
increasing_pairs = [(x, y)
                    for x in range(10) 
                        for y in range(x + 1, 10)]

'''
increasing_pairs = [(u, v)
                    for u in range(v+1, 10) 
                        for v in range(10)]
'''
                         
#####################################################################
# Generator                              
def gen_num(n):
    """a lazy version of range"""
    i = 0
    while i < n:
        yield i
        i += 1

for i in gen_num(10):
    print(i)

def list_num(n):
    return [i for i in range(n)]

for i in list_num(10):
    print(i)
    

l1 = list_num(50000000) 
l2 = gen_num(50000000) #faster!

l1[100]
l2.__next__()
l2[100]     # Error



#####################################################################
# Random
import random
rnums = [random.random() for _ in range(4)]

random.seed(10) # set the seed to 10
print(random.random())
random.seed(10) # reset the seed to 10
print(random.random()) 

random.randrange(10) # choose randomly from range(10) = [0, 1, ..., 9]
random.randrange(3, 6)

up_to_ten = [x for x in range(10)]
random.shuffle(up_to_ten)
print(up_to_ten)


my_best_friend = random.choice(["Alice", "Bob", "Charlie"])

lottery_numbers = range(60)
winning_numbers = random.sample(lottery_numbers, 6)
print(winning_numbers)

# Random : Normal distribution
import numpy as np
mu, sigma = 100, 10
s = np.random.normal(mu, sigma, 100)

sorted(list(s))


#####################################################################
#Regular Expressions
import re

print(all([ # all of these are true, because
    not re.match("a", "cat"), # * 'cat' doesn't start with 'a'
    re.search("a", "cat"), # * 'cat' has an 'a' in it
    not re.search("c", "dog"), # * 'dog' doesn't have a 'c' in it
    3 == len(re.split("[ab]", "carbs")), # * split on a or b to ['c','r','s']
    "R-D-" == re.sub("[0-9]", "-", "R2D2") # * replace digits with dashes
])) # prints

ex = re.compile("[0-9a-z]+", re.I)
ex.match('1234')
ex.match('1abc234')
ex.match('1abcABC234')
ex.match('!')
ex.match('abc!')
ex.match('abc!def')
ex.match('!abc!')
ex.search('!abc!')

#####################################################################
#Enumerations

doc_text = """2개 아상의 DL 분석서버로 구성된 경우,  
    두 서버 사이에 집계 결과가 다를 수 있다. (약 0.05% 이하이긴하나...) 이 경우, 
    보고서를 보는 사람에 따라 운이 없으면 집계 수치가 감소하는 현상이 생길 수도 있다."""

doc = doc_text.split()
#doc = doc_text.split('.')

for  w in doc:
    print(w)
    
for i, w in enumerate(doc):
    print("%d : %s" % (i, w))

for _, w in enumerate(doc):
    print(w)
    
for i, _ in enumerate(doc):
    print(i)
    
    

#####################################################################
# function tools, partial
def exp(base, power):
    return base ** power

def two_to_the(power):
    return exp(2, power)

print(two_to_the(4))


from functools import partial
two_to_the = partial(exp, 2) # is now a function of one variable
print(two_to_the(3)) # 8


square_of = partial(exp, power=2)
print(square_of(3) )



#####################################################################
# map & reduce
def double(x):
    return 2 * x

xs = [1, 2, 3, 4]
twice_xs = [double(x) for x in xs] # [2, 4, 6, 8]
twice_xs = map(double, xs) # same as above

list_doubler = partial(map, double) # *function* that doubles a list
twice_xs = list_doubler(xs) # again [2, 4, 6, 8]

def multiply(x, y): return x * y
products = map(multiply, [1, 2], [4, 5]) # [1 * 4, 2 * 5] = [4, 10]


def is_even(x):
    """True if x is even, False if x is odd"""
    return x % 2 == 0

x_evens = [x for x in xs if is_even(x)] # [2, 4]
x_evens = filter(is_even, xs) # same as above
list_evener = partial(filter, is_even) # *function* that filters a list
x_evens = list_evener(xs) # again [2, 4]

from functools import reduce
x_product = reduce(multiply, xs) # = 1 * 2 * 3 * 4 = 24
list_product = partial(reduce, multiply) # *function* that reduces a list
x_product = list_product(xs) # again = 24



#####################################################################
# zip and Argument Unpacking

list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]
list(zip(list1, list2)) # is [('a', 1), ('b', 2), ('c', 3)]

pairs = [('a', 1), ('b', 2), ('c', 3)]
letters, numbers = zip(*pairs)

list(zip(('a', 1), ('b', 2), ('c', 3)))   # [('a','b','c'), ('1','2','3')]


# argument unpacking
def add(a, b): return a + b
add(1, 2) # returns 3
add([1, 2]) # TypeError!
add(*[1, 2]) # returns 3



#####################################################################
# args and kwargs
def doubler(f):
    def g(x):
        return 2 * f(x)
    return g

def f1(x):
    return x + 1

g = doubler(f1)
print(g(3)) # 8 (== ( 3 + 1) * 2)
print(g(-1)) # 0 (== (-1 + 1) * 2)



def f2(x, y):
    return x + y

g = doubler(f2)
print(g(1, 2)) # TypeError: g() takes exactly 1 argument (2 given)

# args is a tuple of its unnamed arguments and kwargs is a dict of its named arguments.
def magic(*args, **kwargs):
    print("unnamed args:", args)
    print("keyword args:", kwargs)
    
magic(1, 2, key="word", key2="word2")
# unnamed args: (1, 2)
# keyword args: {'key2': 'word2', 'key': 'word'}


def other_way_magic(x, y, z):
    return x + y + z

x_y_list = [1, 2]
z_dict = { "z" : 3 }
print(other_way_magic(*x_y_list, **z_dict)) # 6


def doubler_correct(f):
    """works no matter what kind of inputs f expects"""
    def g(*args, **kwargs):
        """whatever arguments g is supplied, pass them through to f"""
        return 2 * f(*args, **kwargs)
    return g

g = doubler_correct(f2)
print(g(1, 2)) # 6
