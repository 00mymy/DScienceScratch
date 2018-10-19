# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:21:54 2017

@author: 00mymy
"""

import math
import random
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from functools import  reduce


##########################################################################
# one-dimensional
#
def bucketize(point, bucket_size):
    """floor the point to the next lower multiple of bucket_size"""
    return bucket_size * math.floor(point / bucket_size)

def make_histogram(points, bucket_size):
    """buckets the points and counts how many in each bucket"""
    return Counter(bucketize(point, bucket_size) for point in points)

def plot_histogram(points, bucket_size, title=""):
    histogram = make_histogram(points, bucket_size)
    plt.bar(list(histogram.keys()), list(histogram.values()), width=bucket_size)
    plt.title(title)
    plt.show()


uniform = [200 * np.random.uniform() - 100 for _ in range(10000)]
normal = [57 * np.random.normal() for _ in range(10000)]

np.mean(uniform)
np.std(uniform)
np.mean(normal)
np.std(normal)


plot_histogram(uniform, 10, "Uniform Histogram")
plot_histogram(normal, 10, "Normal Histogram")

plt.boxplot([uniform, normal],  0, 'bD', showmeans=True)


##########################################################################
# two-dimensional (multi-dimensional)
#
xs = [np.random.normal() for _ in range(1000)]
ys1 = [ x + np.random.normal() / 2 for x in xs]
ys2 = [-x + np.random.normal() / 2 for x in xs]

# histograms 
plot_histogram(ys1, .1, "ys1")
plot_histogram(ys2, .1, "ys2")
plt.boxplot([ys1, ys2],  0, 'bD', showmeans=True)

# scatter
plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
plt.scatter(xs, ys2, marker='.', color='gray', label='ys2')
plt.xlabel('xs')
plt.ylabel('ys')
plt.legend(loc=9)
plt.title("Very Different")
plt.show()

# correlations
np.corrcoef(xs, ys1)
np.corrcoef(xs, ys2)
np.corrcoef(ys1, ys2)

# correlation matrix
np.corrcoef(xs, y=[ys1, ys2])

# scatter matrix
from pandas import DataFrame
from pandas.tools.plotting import scatter_matrix
df = DataFrame(list(zip(xs,ys1,ys2)), columns=['xs', 'ys1', 'ys2'])
scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='hist')



##########################################################################
# Cleaning and Munging
#
    
def try_or_none(f):
    """wraps f to return None if f raises an exception
    assumes f takes only one input"""
    def f_or_none(x):
        try: return f(x)
        except: return
    return f_or_none

def parse_row(input_row, parsers):
    return [try_or_none(parser)(value) if parser is not None else value
            for value, parser in zip(input_row, parsers)]
                
def parse_rows_with(reader, parsers):
    """wrap a reader to apply the parsers to each of its rows"""
    for row in reader:
        yield parse_row(row, parsers)
                



'''
import csv

with open("comma_delimited_stock_prices.txt", "rt") as f:
    reader = csv.reader(f)
    for line in parse_rows_with(reader, [dateutil.parser.parse, None, float]):
        data.append(line)
'''
import dateutil.parser

reader = [
        ['6/20/2014','AAPL','90.91'],
        ['6/20/2014','MSFT','41.68'],
        ['6/20/3014','FB','64.5'],
        ['6/19/2014','AAPL','91.86'],
        ['6/19/2014','MSFT','n/a'],
        ['6/19/2014','FB','64.34'],
    ]

data = []

#date string to date obj., number string to float('n/a' noise to None)
for line in parse_rows_with(reader, [dateutil.parser.parse, None, float]):
    data.append(line)

for row in data:
    if any(x is None for x in row):
        print(row)
        

'''
def try_parse_field(field_name, value, parser_dict):
    """try to parse value using the appropriate function from parser_dict"""
    parser = parser_dict.get(field_name) # None if no such entry
    if parser is not None:
        return try_or_none(parser)(value)
    else:
        return value

def parse_dict(input_dict, parser_dict):
    return { field_name : try_parse_field(field_name, value, parser_dict)
             for field_name, value in input_dict.items() }
'''                 

##########################################################################
# Manipulating Data
#

#data_dict = [{"date":row[0], "symbol":row[1], "closing_price":row[2]} for row in data]    
data_dict = [{"date":row[0], "symbol":row[1], "closing_price":row[2] if row[2] != None else 0.01} for row in data]    

max_aapl_price = max(row["closing_price"]
                        for row in data_dict
                            if row["symbol"] == "AAPL")


by_symbol = defaultdict(list)
for row in data_dict:
    by_symbol[row["symbol"]].append(row)

max_price_by_symbol = { symbol : max(row["closing_price"]
    for row in grouped_rows)
        for symbol, grouped_rows in dict(by_symbol).items() }

def picker(field_name):
    """returns a function that picks a field out of a dict"""
    return lambda row: row[field_name]

def pluck(field_name, rows):
    """turn a list of dicts into the list of field_name values"""
    return map(picker(field_name), rows)

def group_by(grouper, rows, value_transform=None):
    # key is output of grouper, value is list of rows
    grouped = defaultdict(list)
    for row in rows:
        grouped[grouper(row)].append(row)
    if value_transform is None:
        return grouped
    else:
        return { key : value_transform(rows)
                 for key, rows in grouped.items() }
        
def percent_price_change(yesterday, today):
    return today["closing_price"] / yesterday["closing_price"] - 1

def day_over_day_changes(grouped_rows):
    # sort the rows by date
    ordered = sorted(grouped_rows, key=picker("date"))
    # zip with an offset to get pairs of consecutive days
    return [{ "symbol" : today["symbol"],
             "date" : today["date"],
             "change" : percent_price_change(yesterday, today) }
                for yesterday, today in zip(ordered, ordered[1:])]

changes_by_symbol = group_by(picker("symbol"), data_dict, day_over_day_changes)
all_changes = [change
                   for changes in changes_by_symbol.values()
                       for change in changes]

max(all_changes, key=picker("change"))
min(all_changes, key=picker("change"))


def combine_pct_changes(pct_change1, pct_change2):
    return (1 + pct_change1) * (1 + pct_change2) - 1

def overall_change(changes):
    return reduce(combine_pct_changes, pluck("change", changes))

overall_change_by_month = group_by(lambda row: row['date'].month, all_changes, overall_change)

#
# Using Pandas DataFrame
pdf = DataFrame(data_dict)
gs = pdf.groupby('symbol')
gs['closing_price'].max()
gs.get_group('AAPL')
gs.groups
gs.describe()
gs.boxplot()

for name, group in gs:
    print(name)
    print(group)

gsd = pdf.groupby(['symbol', 'date'])
gsd['closing_price'].max()




#
# Principal Component Analysis (PCA)
#

#from mpl_toolkits.mplot3d import Axes3D


# Create a signal with only 2 useful dimensions
x1 = np.random.normal(size=100)
x2 = np.random.normal(size=100)
x3 = x1 + x2    #x3 is not useful (depends on x1, x2)
X = np.c_[x1, x2, x3]

''''
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, x3)
'''

from sklearn import decomposition
pca = decomposition.PCA()
pca.fit(X)

print(pca.explained_variance_)  # high variance means more importance
# As we can see, only the 2 first components are useful

pca.n_components = 2
X_reduced = pca.fit_transform(X)
X_reduced.shape
