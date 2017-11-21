# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 18:10:07 2017

@author: Rebecca
"""
import numpy as np

def confusion_matrix(y_true, y_pred):
    tp,fp,tn,fn = 0,0,0,0
    for i in range(0,len(y_pred)):
        if y_pred[i]:
            if y_true[i]:
                tp=tp+1
            else:
                fp=fp+1
        else:
            if y_true[i]:
                fn=fn+1
            else:
                tn=tn+1
    
    confM = np.matrix([[tp,fp],[fn,tn]])
    return confM