# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:46:11 2017

@author: Rebecca
"""

from computarCusto import computarCusto

def gd(x,y,theta,alpha):
    maxiterations = 100000
    precision = 0.00001
    converged = False
    
    iteration = 0
    
    Jtheta, grad = computarCusto(x,y,theta)
    
    while not converged and iteration < maxiterations:
        iteration += 1
        for j in range(len(theta)):
            theta[j] = theta[j] - alpha * grad[j]
        
        new_Jtheta, new_grad = computarCusto(x,y,theta)
        step = Jtheta - new_Jtheta
        Jtheta, grad = new_Jtheta, new_grad 
        if step < 0:
            raise Exception('Increased cost. New Jtheta was bigger than previous Jtheta.')
        if step > 0 and step <= precision:
            converged = True
    
    return theta,Jtheta,converged