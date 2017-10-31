# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 16:43:41 2017

@author: Rebecca
"""

import numpy as np
from sigmoide import sigmoide

def computarCusto(x,y,theta):
    x = np.matrix(x)
    y = np.matrix(y)
    theta = np.matrix(theta)
    m = len(x)
    
    first_term = np.multiply(-y, np.log(htheta(x,theta)))
    second_term = np.multiply((1 - y), np.log(1 - htheta(x,theta)))
    
    JTheta = np.sum(first_term - second_term) / m
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    for j in range(parameters):
        grad[j] = np.sum( np.multiply((htheta(x,theta) - y), x[:,j]) ) / m
        
    return JTheta, grad

def htheta(x,theta):
    return sigmoide( x * theta.T )