# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 15:35:11 2017

@author: Rebecca
"""

import numpy as np
from gdmulti import gd
from normalizarCaracteristica import normalizarCaracteristica
from importarVariavel import importarDados

filepath = "\ex1data2.txt"
x,y = importarDados(filepath,["Size","Dorms","Price"])

theta = np.zeros(3)
alpha = 0.01

x_norm,par_norm = normalizarCaracteristica(x)

theta,Jtheta,converged = gd(x_norm,y,theta,alpha)