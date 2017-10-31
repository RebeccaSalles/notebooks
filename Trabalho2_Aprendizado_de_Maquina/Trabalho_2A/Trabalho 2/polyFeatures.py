# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:07:52 2017

@author: Rebecca
"""

import numpy as np

def polyFeatures(x, degree=8):
    m = len(x)

    featureMap = np.empty(shape=(m, degree+1), dtype=float)
    
    col = 0
    for i in xrange(0,degree+1):
        featureMap[:,col] = x[:,1]**i
        col += 1
    
    featureMap = np.array(np.matrix(featureMap))
    
    return featureMap