# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 16:04:06 2017

@author: Rebecca
"""

import numpy as np
import matplotlib.pyplot as plt
from importarVariavel import importarDados

filepath = "\ex2data1.txt"
x,y = importarDados(filepath,['Exam 1', 'Exam 2', 'Admitted'])

theta = np.zeros(3)
alpha = 0.01

x_a = [xi[np.where(y == 1)[0]] for xi in x.T]
x_na = [xi[np.where(y == 0)[0]] for xi in x.T]


adm, = plt.plot(x_a[1], x_a[2], "+" ,c="black", label="Admitted")
nadm, = plt.plot(x_na[1], x_na[2], "o", c="yellow", label="Not admitted")
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(handles=[adm, nadm], loc='upper right')
plt.show()