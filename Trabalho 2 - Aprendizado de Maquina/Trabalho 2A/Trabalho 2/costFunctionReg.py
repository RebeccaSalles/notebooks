# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:55:31 2017

@author: Rebecca
"""

import numpy as np
from sigmoide import sigmoide

def costFunctionReg(x,y,theta,lamda):
    x = np.matrix(x)
    y = np.matrix(y)
    theta = np.matrix(theta)
    m = len(x)
    
    first_term = np.multiply(-y, np.log(htheta(x,theta)))
    second_term = np.multiply((1 - y), np.log(1 - htheta(x,theta)))
    regularization_term = (float(lamda)/(2.0*m)) * np.sum(np.power(theta[:,1:],2))

    
    JTheta = np.sum(first_term - second_term) / m + regularization_term
    
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    
    for j in range(parameters):
        regularization_term = 0 if j == 0 else (float(lamda)/m)*theta[:,j]
        grad[j] = np.sum( np.multiply((htheta(x,theta) - y), x[:,j]) ) / m + regularization_term
    
    return JTheta, grad

def htheta(x,theta):
    return sigmoide( x * theta.T )


#from mapFeature import mapFeature
#from importarVariavel import importarDados
#filepath = "\ex2data2.txt"
#x,y = importarDados(filepath,['Microship Test 1', 'Microship Test 2', 'Aceito'])
#x_tmp = mapFeature(x[:,1],x[:,2],18)
#costFunctionReg(x_tmp,y,np.zeros(190),1)