# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 20:22:16 2017

@author: Rebecca
"""
import numpy as np

def normalizarCaracteristica(x, mu=None, sigma=None):
    m = x.shape[0]

    if mu is None:
        mu = np.mean(x, axis=0)
    if sigma is None:
        sigma = np.std(x, axis=0, ddof=1)

    # don't change the intercept term
    mu[0] = 0.0
    sigma[0] = 1.0

    for i in range(m):
        x[i, :] = (x[i, :] - mu) / sigma

    return x, mu, sigma

#normalizarCaracteristica(x)
#x_norm.T[1] == (x[1]-np.mean(x[1]))/np.std(x[1])