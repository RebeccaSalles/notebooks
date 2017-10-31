# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:47:50 2017

@author: Rebecca
"""

import numpy as np
import matplotlib.pyplot as plt
from gduni import gd
from computarCusto import htheta
from importarVariavel import importarDados

filepath = "\ex1data1.txt"
x,y = importarDados(filepath,["Population","Profit"])

theta = np.zeros(2)
alpha = 0.01

theta,Jtheta,converged = gd(x,y,theta,alpha)

print("GD converged: " + str(converged))

train, = plt.plot(x[:,1], y, "x", c="red", label="Training data")
reg, = plt.plot(x[:,1],htheta(x,np.matrix(theta)), 'k' ,c="blue", label="Linear regression")
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.legend(handles=[train, reg], loc='lower right')
plt.show()