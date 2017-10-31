# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 15:47:42 2017

@author: Rebecca
"""

import numpy as np

def selectThreshold(prXval,yval):
    epsilon = 0
    f1 = 0
    
    step = (prXval.max() - prXval.min()) / 1000
    
    for curr_e in np.arange(prXval.min(), prXval.max(), step):
        tp,fp,fn = 0.0,0.0,0.0
        for i in range(prXval.shape[0]):
            pred = prXval[i] < curr_e
            y = yval[i]
            tp = tp+1 if pred == 1 and y == 1 else tp
            fp = fp+1 if pred == 1 and y == 0 else fp
            fn = fn+1 if pred == 0 and y == 1 else fn
        
        prec = 0 if tp == 0 else tp/(tp + fp)
        rec = 0 if tp == 0 else tp/(tp + fn)
        curr_f1 = 0 if prec+rec == 0 else (2*prec*rec)/(prec+rec)
        
        if curr_f1 > f1:
            f1 = curr_f1
            epsilon = curr_e
    
    return epsilon, f1    