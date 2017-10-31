# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 19:05:49 2017

@author: Rebecca
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from computarCusto import computarCusto
from mpl_toolkits.mplot3d import Axes3D
from importarVariavel import importarDados

filepath = "\ex1data1.txt"
x,y = importarDados(filepath,["Population","Profit"])

theta0 = np.arange(-10, 10.01, 0.01)
theta1 = np.arange(-1, 4.01, 0.01)

J = [ [computarCusto(x,y,[t0,t1])[0] for t0 in theta0] for t1 in theta1]

theta0, theta1 = np.meshgrid(theta0, theta1)

lvls = np.arange(0, 750, 10)
colorbar_ticks = np.arange(0, 800, 50)
CS = plt.contourf(theta0, theta1, J, levels=lvls, cmap=cm.spectral)
plt.colorbar(CS, shrink=1, extend='both', ticks=colorbar_ticks)
plt.title("Custo J(Theta) vs. Parametros Theta0 e Theta1")
plt.xlabel("Theta 0")
plt.ylabel("Theta 1")
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(theta0, theta1, J, cmap=cm.spectral)
plt.colorbar(surf, shrink=0.8, extend='both', ticks=colorbar_ticks)
plt.title("Custo J(Theta) vs. Parametros Theta0 e Theta1")
plt.xlabel("Theta 0")
plt.ylabel("Theta 1")
plt.show()