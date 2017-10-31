# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 12:18:48 2017

@author: Rebecca
"""

import numpy as np
import matplotlib.pyplot as plt
from gaussianModel import probModel as P
import scipy.io as spio

mat = spio.loadmat('ex8data1.mat', squeeze_me=True)

X = mat['X']

p,mu,sigma2 = P(X,None,None)

data, = plt.plot(X[:,0], X[:,1], "x", c="blue", label="Data")

#z = np.zeros([50, 50])
#x1 = np.linspace(0.0, 30.0, 50)
#x2 = np.linspace(0.0, 30.0, 50)
#for i, t1 in enumerate(x1):
#    for j, t2 in enumerate(x2): 
#        z[i, j] = P(np.matrix([[t1,t2]]),mu,sigma2)[0]
#c = plt.contour(x1, x2, z.T, [0], colors='dodgerblue')
#c.collections[0].set_label("Gaussian contour")

plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
#plt.title('Contours of Gaussian Distribution')
#plt.legend(loc='upper right')
#plt.savefig('plot1.2.png')
plt.show()