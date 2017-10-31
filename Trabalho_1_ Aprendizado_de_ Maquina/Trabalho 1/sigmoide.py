# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 16:29:33 2017

@author: Rebecca
"""

import numpy as np

def sigmoide(z):
    return 1.0 / (1 + np.exp(-z))