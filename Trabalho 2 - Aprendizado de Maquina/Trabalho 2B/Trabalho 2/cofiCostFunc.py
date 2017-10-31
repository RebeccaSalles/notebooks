# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 20:50:18 2017

@author: Rebecca
"""

import numpy as np

def cofiCostFunc(params, Y, R, nFeatures):  
    Y = np.matrix(Y)
    R = np.matrix(R)
    nm = Y.shape[0]
    nu = Y.shape[1]
    
    X = np.matrix(np.reshape(params[:(nm*nFeatures)], (nm, nFeatures)))
    Theta = np.matrix(np.reshape(params[(nm*nFeatures):], (nu, nFeatures)))
    
    error = np.multiply(htheta(X,Theta) - Y, R)
    J = np.sum( np.power( error , 2) ) / (2.0)
    
    X_grad = np.dot(error,Theta)
    Theta_grad = np.dot(error.T,X)
    
    grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))
    
    return J, grad

def htheta(X,Theta):
    return np.dot(X, Theta.T)