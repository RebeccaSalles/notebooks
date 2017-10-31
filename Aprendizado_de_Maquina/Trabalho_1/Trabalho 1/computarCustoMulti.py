# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 15:24:24 2017

@author: Rebecca
"""

import numpy as np

def computarCusto(x,y,theta):
    x = np.matrix(x)
    y = np.matrix(y)
    theta = np.matrix(theta)
    m = len(x)
    
    JTheta = np.sum( np.power((htheta(x,theta) - y),2) ) / (2.0*m)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    for j in range(parameters):
        grad[j] = np.sum( np.multiply((htheta(x,theta) - y), x[:,j]) ) / m

    return JTheta, grad

def htheta(x,theta):
    return x * theta.T