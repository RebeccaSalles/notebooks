# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 15:27:00 2017

@author: Rebecca
"""

import matplotlib.pyplot as plt
from importarVariavel import importarDados

filepath = "\ex1data1.txt"
x,y = importarDados(filepath,["Population","Profit"])

plt.plot(x[:,1], y, "x", c="red")
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()