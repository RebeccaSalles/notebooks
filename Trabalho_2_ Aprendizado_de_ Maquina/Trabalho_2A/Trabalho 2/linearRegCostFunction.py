# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 20:50:18 2017

@author: Rebecca
"""

import numpy as np

def linearRegCostFunction(x,y,theta,lamda):
    x = np.matrix(x)
    y = np.matrix(y).T
    #theta = np.matrix(theta)
    m = len(x)
    
    regularization_term = (float(lamda)/(2*m)) * np.sum(np.power(theta[1:theta.shape[0]],2))
    
    JTheta = np.sum( np.power((htheta(x,theta) - y),2) ) / (2.0*m) + regularization_term
    
    parameters = int(theta.ravel().shape[0])
    grad = np.zeros(parameters)

    for j in range(parameters):
        regularization_term = 0 if j == 0 else (float(lamda)/m)*theta[j]
        grad[j] = np.sum( np.multiply((htheta(x,theta) - y), x[:,j]) ) / m + regularization_term

    return JTheta, grad

def htheta(x,theta):
    return np.dot(x, theta)

#X = np.array(np.insert(np.matrix(X),[0],np.matrix(np.ones(12)),axis=0)).T
#y = np.array(np.matrix(y).T)
#linearRegCostFunction(x,y,np.ones(2),1)