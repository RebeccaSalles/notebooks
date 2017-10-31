# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 12:11:32 2017

@author: Rebecca
"""

import numpy as np
from estimativaGaussian import estimativaGaussian

def gaussian(x,mu,sigma2):
    return 1./(np.sqrt(2*np.pi*sigma2)) * np.exp(-np.power((x - mu), 2)/(2*sigma2))

def probModel(x,mu=None,sigma2=None):
    var = x.shape[1]
    
    if mu is None or sigma2 is None:
        mu,sigma2 = estimativaGaussian(x)
    
    p = []
    for j in range(var):
        p.append( gaussian(x[:, j],mu[j],sigma2[j]) )
    
    p = np.prod(p, axis=0)
    
    return p,mu,sigma2