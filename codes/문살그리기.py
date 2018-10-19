# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:12:59 2017

@author: 00mymy
"""
from matplotlib import pyplot as plt

num_points = 73
#num_points = 21
xs = [x*40 for x in range(num_points)] # 2800 / 38
ys = [0 for x in range(num_points)]

for i in range(int(800/38)):
    ys[i] = 2400 if (xs[i]/40)%2==0 else 1200*(.66**int(xs[i]/40/2))
# [2400 if (x/40)%2==0 else 1200*(.66**int(x/40/2)) for x in xs] 


plt.bar(xs, ys, 40)
plt.axis([0, 2800, 0, 2400])

plt.gca().set_aspect('equal', adjustable='box')
plt.show()

#plt.savefig('문살모양')