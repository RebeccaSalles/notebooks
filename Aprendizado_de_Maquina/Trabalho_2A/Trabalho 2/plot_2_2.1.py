# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 20:40:07 2017

@author: Rebecca
"""

import matplotlib.pyplot as plt
import scipy.io as spio

mat = spio.loadmat('ex5data1.mat', squeeze_me=True)

X = mat['X']
Xtest = mat['Xtest']
Xval = mat['Xval']

y = mat['y']
ytest = mat['ytest']
yval = mat['yval']

plt.plot(X, y, "x", c="red")
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
#plt.savefig('plot1.1.png')
plt.show()