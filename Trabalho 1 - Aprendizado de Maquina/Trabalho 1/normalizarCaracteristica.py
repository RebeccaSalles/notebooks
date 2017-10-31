# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 20:22:16 2017

@author: Rebecca
"""
import numpy as np

def normalizarCaracteristica(x):
    x = x.T
    x_norm = [ (xi - np.mean(xi))/np.std(xi) if np.std(xi)!= 0 else xi for xi in x]
    x_norm = np.array(np.matrix(x_norm).T)
    par_norm = [ {'mean':np.mean(xi),'std':np.std(xi)} for xi in x]
    
    return x_norm, par_norm

#normalizarCaracteristica(x)
#x_norm.T[1] == (x[1]-np.mean(x[1]))/np.std(x[1])