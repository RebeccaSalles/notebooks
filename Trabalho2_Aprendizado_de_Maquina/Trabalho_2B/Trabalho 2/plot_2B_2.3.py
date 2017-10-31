# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 17:08:09 2017

@author: Rebecca
"""

import numpy as np
import matplotlib.pyplot as plt
from gaussianModel import probModel as P
from selectThreshold import selectThreshold
import scipy.io as spio

mat = spio.loadmat('ex8data1.mat', squeeze_me=True)

X = mat['X']
Xval = mat['Xval']  
yval = mat['yval']

p,mu,sigma2 = P(X,None,None)

prXval = P(Xval,mu,sigma2)[0]

epsilon, f1 = selectThreshold(prXval,yval)
print 'Selected threshold: ' + str(epsilon) + ', F1: ' + str(f1)

outls = np.where(p < epsilon)[0]

data, = plt.plot(X[:,0], X[:,1], "x", c="blue", label="Data")
outliers, = plt.plot(X[outls,0], X[outls,1], "o", c="red", label="Outliers")

plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.legend(loc='upper right')
#plt.savefig('plot1.2.png')
plt.show()