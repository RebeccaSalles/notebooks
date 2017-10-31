# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 16:04:06 2017

@author: Rebecca
"""

import numpy as np
import matplotlib.pyplot as plt
from importarVariavel import importarDados

filepath = "\ex2data2.txt"
x,y = importarDados(filepath,['Microship Test 1', 'Microship Test 2', 'Aceito'])

x_a = [xi[np.where(y == 1)[0]] for xi in x.T]
x_na = [xi[np.where(y == 0)[0]] for xi in x.T]

adm, = plt.plot(x_a[1], x_a[2], "+", c="black", label="y = 1")
nadm, = plt.plot(x_na[1], x_na[2], "o", c="yellow", label="y = 0")
plt.xlabel('Microship Test 1')
plt.ylabel('Microship Test 2')
plt.legend(handles=[adm, nadm], loc='upper right')
plt.show()