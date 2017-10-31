# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 11:41:45 2017

@author: Rebecca
"""

import numpy as np

def estimativaGaussian(x):
    m = x.shape[0]
    var = x.shape[1]

    mu = []
    sigma2 = []
    for j in range(var):
        mu.append(np.sum(x[:, j]) / float(m))
        sigma2.append(np.sum( np.power( (x[:, j] - mu[j]) ,2) ) / float(m))
    
    return np.array(mu), np.array(sigma2)