# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:43:35 2017

@author: Rebecca
"""

import numpy as np
import matplotlib.pyplot as plt
from linearRegGD import gd
from linearRegCostFunction import htheta
from normalizarCaracteristica import normalizarCaracteristica
from polyFeatures import polyFeatures

import scipy.io as spio

mat = spio.loadmat('ex5data1.mat', squeeze_me=True)

X = mat['X']
y = mat['y']
x = np.array(np.insert(np.matrix(X),[0],np.matrix(np.ones(len(X))),axis=0)).T
x_poly = polyFeatures(x,8)
x_norm, mu, sigma = normalizarCaracteristica(x_poly)
y = np.array(y)

theta = np.ones((x_norm.shape[1],1))
alpha = 1e-2
lamda = 0

theta,Jtheta,converged,iterations = gd(x_norm,y,theta,alpha,lamda)
print("GD converged: " + str(converged) + ", Iterations: " + str(iterations))


x_train = np.array(np.matrix(np.arange(min(x[:,1]) - 15, max(x[:,1]) + 25, 0.05)))
x_train = np.array(np.insert(np.matrix(x_train),[0],np.matrix(np.ones(len(x_train))),axis=0)).T
x_poly = polyFeatures(x_train,8)
x_norm, mu, sigma = normalizarCaracteristica(x_poly, mu, sigma)
y_train = htheta(x_norm,np.matrix(theta))
reg, = plt.plot(x_train[:,1], y_train, '--' ,c="blue", label="Polynomial regression")
train, = plt.plot(x[:,1], y, "x", c="red", label="Training data")
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial regression (lambda = 0.0)')
plt.legend(handles=[train, reg], loc='lower right')
#plt.savefig('plot1.2.png')
plt.show()