# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:45:09 2017

@author: Rebecca
"""

import numpy as np
import matplotlib.pyplot as plt
from linearRegGD import gd
from linearRegCostFunction import htheta

import scipy.io as spio

mat = spio.loadmat('ex5data1.mat', squeeze_me=True)

X = mat['X']
y = mat['y']
x = np.array(np.insert(np.matrix(X),[0],np.matrix(np.ones(len(X))),axis=0)).T
y = np.array(y)

theta = np.ones((x.shape[1],1))
alpha = 1e-3
lamda = 0

theta,Jtheta,converged,iterations = gd(x,y,theta,alpha,lamda)
print("GD converged: " + str(converged) + ", Iterations: " + str(iterations))

train, = plt.plot(x[:,1], y, "x", c="red", label="Training data")
reg, = plt.plot(x[:,1],htheta(x,np.matrix(theta)), 'k' ,c="blue", label="Linear regression")
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
#plt.legend(handles=[train, reg], loc='lower right')
#plt.savefig('plot1.2.png')
plt.show()