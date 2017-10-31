# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:43:47 2017

@author: Rebecca
"""

import numpy as np
from linearRegCostFunction import linearRegCostFunction as computarCusto
from linearRegGD import gd

def learningCurve(x,y,xval,yval,initialTheta,alpha,lamda):
    trainingError = []
    valError = []
    for i in range(1,len(x)):
        x_tmp = x[:i,:]
        y_tmp = y[:i]
        theta,Jtheta,converged,iterations = gd(x_tmp,y_tmp,initialTheta,alpha,lamda)
        if not converged:
            raise Exception('GD with ' + str(i) + ' training examples did not converged.')
        errorT,grad = computarCusto(x_tmp,y_tmp,theta,0)
        trainingError.append(errorT)
        errorVal,grad = computarCusto(xval,yval,theta,0)
        valError.append(errorVal)
    
    return np.array(trainingError),np.array(valError)