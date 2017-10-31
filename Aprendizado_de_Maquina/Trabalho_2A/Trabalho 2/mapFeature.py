# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:31:27 2017

@author: Rebecca
"""

import numpy as np

def mapFeature(x1,x2, degree=6):
    m = len(x1)
    
    num_features = (degree + 1) * (degree + 2) / 2
    featureMap = np.empty(shape=(m, num_features), dtype=float)
    
    col = 0
    for i in xrange(0,degree+1):
        for j in xrange(0,i+1):
            featureMap[:,col] = x1**(i - j) * x2**j
            col+=1
    
    featureMap = np.array(np.matrix(featureMap))
    
    return featureMap