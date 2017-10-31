# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:27:38 2017

@author: Rebecca
"""

import numpy as np
import matplotlib.pyplot as plt
from gdReg import gd
from mapFeature import mapFeature
from importarVariavel import importarDados

filepath = "\ex2data2.txt"
x,y = importarDados(filepath,['Microship Test 1', 'Microship Test 2', 'Aceito'])

x = mapFeature(x[:,1],x[:,2])
theta = np.zeros(x.shape[1])
alpha = 0.01
lamda = 1

theta,Jtheta,converged,iterations = gd(x,y,theta,alpha,lamda)
print("GD converged: " + str(converged) + ", Iterations: " + str(iterations))


#Plotting decision boundary
x_a = [xi[np.where(y == 1)[0]] for xi in x.T]
x_na = [xi[np.where(y == 0)[0]] for xi in x.T]

adm, = plt.plot(x_a[1], x_a[2], "+", c="black", label="y = 1")
nadm, = plt.plot(x_na[1], x_na[2], "o", c="yellow", label="y = 0")

z = np.zeros([50, 50])
test1 = np.linspace(-1.0, 1.5, 50)
test2 = np.linspace(-1.0, 1.5, 50)
for i, t1 in enumerate(test1):
    for j, t2 in enumerate(test2): 
        z[i, j] = np.dot(mapFeature(np.array([[t1]]),np.array([[t2]])), np.matrix(theta).T)[0]
db = plt.contour(test1, test2, z.T, [0], colors='dodgerblue')
db.collections[0].set_label("Decision boundary")

plt.xlabel('Microship Test 1')
plt.ylabel('Microship Test 2')
plt.title('Decision boundary for lambda = ' + str(lamda))
plt.legend(loc='upper right')
plt.show()