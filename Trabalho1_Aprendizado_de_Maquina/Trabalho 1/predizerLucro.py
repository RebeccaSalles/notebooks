# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 18:41:44 2017

@author: Rebecca
"""

import numpy as np
from gduni import gd
from computarCusto import htheta
from importarVariavel import importarDados

filepath = "\ex1data1.txt"
x,y = importarDados(filepath,["Population","Profit"])

theta = np.zeros(2)
alpha = 0.01

theta,Jtheta,converged = gd(x,y,theta,alpha)

#Since the dataset is in 10,000s, 35,000 and 70,000 habitants are represented as 3.5 and 7.0
test_set = np.array(np.matrix([[1.0,3.5],[1.0,7.0]]))
htheta(test_set,np.matrix(theta))