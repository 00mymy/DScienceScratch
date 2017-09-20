# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 18:02:39 2017

@author: 00mymy
"""


from matplotlib import pyplot as plt


'''
years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]
# create a line chart, years on x-axis, gdp on y-axis
plt.plot(years, gdp, color='green', marker='o', linestyle='solid')
#help(plt.plot) :color, marker, linestyle 다양함

# add a title
plt.title("Nominal GDP")
# add a label to the y-axis
plt.ylabel("Billions of $")
plt.show()
'''


'''
movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

#그냥 무대뽀로 그려보기 (안됨)
#plt.bar(movies, num_oscars)
#plt.show()
#help(plt.bar) 해보자
#bar 그래프의 x축에 scalar 값을 줘야한다.

# 그러면...
#plt.bar([1,2,3,4,5], num_oscars)
#plt.show()
#된다. 좀 더 이쁘게 나오도록 다음자

# bars are by default width 0.8, so we'll add 0.1 to the left coordinates
# so that each bar is centered
xs = [i  for i, _ in enumerate(movies)]
#xs = [i + 0.1 for i, _ in enumerate(movies)]
# plot bars with left x-coordinates [xs], heights [num_oscars]
plt.bar(xs, num_oscars)
plt.ylabel("# of Academy Awards")
plt.title("My Favorite Movies")
# label x-axis with movie names at bar centers
#plt.xticks([i for i, _ in enumerate(movies)], movies)
plt.xticks([i + 0.5 for i, _ in enumerate(movies)], movies)
plt.show()
'''

'''
grades = [83,95,91,87,70,0,85,82,100,67,73,77,0]
decile = lambda grade: grade // 10 * 10
histogram = Counter(decile(grade) for grade in grades)
plt.bar([x-4  for x in histogram.keys()], # shift each bar to the left by 4
    histogram.values(), # give each bar its correct height
    8) # give each bar a width of 8
plt.axis([-5, 105, 0, 5]) # x-axis from -5 to 105,
# y-axis from 0 to 5
plt.xticks([10 * i for i in range(11)]) # x-axis labels at 0, 10, ..., 100
plt.xlabel("Decile")
plt.ylabel("# of Students")
plt.title("Distribution of Exam 1 Grades")
plt.show()
'''

'''
from collections import Counter
grades = [83,95,91,87,70,0,85,82,100,67,73,77,0]
decile = lambda grade: grade // 10 * 10
histogram = Counter(decile(grade) for grade in grades)

plt.bar([x-4 for x in histogram.keys()],  # shift each bar to the left by 4
        histogram.values(),                 # give each bar its correct height
        8)                                  # give each bar a width of 8

# 좀 더 보기 좋게하기
#plt.axis([-5, 105, 0, 5])   # x-axis from -5 to 105,
                            # y-axis from 0 to 5
#plt.xticks([10 * i for i in range(11)]) # x-axis labels at 0, 10, ..., 100

plt.xlabel("Decile")
plt.ylabel("# of Students")
plt.title("Distribution of Exam 1 Grades")
plt.show()
'''

'''
mentions = [500, 505]
years = [2013, 2014]

#일단, 그냥 그려보기 (그리긴 그리는데...)
#plt.bar(years,	mentions) 
#plt.xticks(years) 
#plt.ylabel("# of times I heard someone say 'data science'")
#plt.title("Look at the 'Huge' Increase!") 
#plt.show()

plt.bar([2012.6, 2013.6], mentions, 0.8)
#plt.xticks(years)
plt.ylabel("# of times I heard someone say 'data science'")
# if you don't do this, matplotlib will label the x-axis 0, 1

# and then add a +2.013e3 off in the corner (bad matplotlib!)
# 이거 안 넣으면 영 이상함
plt.ticklabel_format(useOffset=False)

# misleading y-axis only shows the part above 500
plt.axis([2012.5,2014.5,499,506])
plt.title("Look at the 'Huge' Increase!")

#스케일 보정
plt.axis([2012.5, 2014.5,0,550])
plt.title("Not So Huge Anymore")
plt.show()

#앞에서 배운거 응용 - 간단하게
plt.bar([.1, 1.1], mentions, .8)
plt.xticks([.5, 1.5], years)
plt.title("Not So Huge Anymore")
plt.show()
'''



'''
var = [1, 2, 4, 8, 16, 32, 64, 128, 256]
bias_sqrd = [256, 128, 64, 32, 16, 8, 4, 2, 1]
total_error = [x + y for x, y in zip(var, bias_sqrd)]
xs = [i for i, _ in enumerate(var)]
plt.plot(xs, var, 'g-', label='variance')
plt.plot(xs, bias_sqrd, 'r-.', label='bias^2')
plt.plot(xs, total_error, 'b:', label='total error') 

# loc=9 means "top center"
# help(plt.legend)
plt.legend(loc=9) 	
plt.xlabel("model complexity")
plt.title("The Bias-Variance Tradeoff")
plt.show()
'''

'''
friends = [ 70, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)
# label each point
for label, friend_count, minute_count in zip(labels, friends, minutes):
    plt.annotate(label,
                 xy=(friend_count, minute_count), # put the label with its point
                 xytext=(5, -5), # but slightly offset
                 textcoords='offset points')
plt.title("Daily Minutes vs. Number of Friends")
plt.xlabel("# of friends")
plt.ylabel("daily minutes spent on the site")

#좀 더 의미있게 보려면
#plt.axis([-5,max(friends)+10,-5,max(minutes)+10])
plt.show()
'''

'''
test_1_grades = [ 99, 90, 85, 97, 80]
test_2_grades = [100, 85, 60, 90, 70]

plt.scatter(test_1_grades, test_2_grades)
plt.title("Axes Aren't Comparable")
plt.xlabel("test 1 grade")
plt.ylabel("test 2 grade")
#plt.axis("equal")
plt.show()
'''

'''
test_1_grades = [85, 60, 90, 70, 99, 90, 85, 97, 80, 120]
test_2_grades = [35, 100, 85, 60, 90, 70, 70, 99, 90, 85, 97]
plt.boxplot([test_1_grades, test_2_grades],  0, 'bD', showmeans=True)
'''

###############################################
# More about Boxplot
###############################################
import numpy as np
#import matplotlib.pyplot as plt

# fake data
np.random.seed(937)
data = np.random.lognormal(size=(37, 4), mean=1.5, sigma=1.75)
#data = np.random.normal(size=(37, 4), loc=1.5, scale=1.75)
labels = list('ABCD')
fs = 10  # fontsize


plt.boxplot(data, labels=labels, showmeans=True, showfliers=False)
plt.show()


# demonstrate how to toggle the display of different elements:
'''
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 6), sharey=True)
axes[0, 0].boxplot(data, labels=labels)
axes[0, 0].set_title('Default', fontsize=fs)

axes[0, 1].boxplot(data, labels=labels, showmeans=True)
axes[0, 1].set_title('showmeans=True', fontsize=fs)

axes[0, 2].boxplot(data, labels=labels, showmeans=True, meanline=True)
axes[0, 2].set_title('showmeans=True,\nmeanline=True', fontsize=fs)

axes[1, 0].boxplot(data, labels=labels, showbox=False, showcaps=False)
tufte_title = 'Tufte Style \n(showbox=False,\nshowcaps=False)'
axes[1, 0].set_title(tufte_title, fontsize=fs)

axes[1, 1].boxplot(data, labels=labels, notch=True, bootstrap=10000)
axes[1, 1].set_title('notch=True,\nbootstrap=10000', fontsize=fs)

axes[1, 2].boxplot(data, labels=labels, showfliers=False)
axes[1, 2].set_title('showfliers=False', fontsize=fs)

for ax in axes.flatten():
    ax.set_yscale('log')
    ax.set_yticklabels([])

fig.subplots_adjust(hspace=.4)
plt.show()
'''


"""
# 3D도 된다.
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
"""

#word cloud
from wordcloud import WordCloud


# Read the whole text.
text = '''Data scientist has been called “the sexiest job of the 21st century,” presumably by
someone who has never visited a fire station. Nonetheless, data science is a hot and
growing field, and it doesn’t take a great deal of sleuthing to find analysts breathlessly
prognosticating that over the next 10 years, we’ll need billions and billions more data
scientists than we currently have.'''

# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

# lower max_font_size
wordcloud = WordCloud(max_font_size=40).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()