# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 17:47:19 2017

@author: Rebecca
"""

import numpy as np
import matplotlib.pyplot as plt
from learningCurve import learningCurve
from normalizarCaracteristica import normalizarCaracteristica
from polyFeatures import polyFeatures
import scipy.io as spio

mat = spio.loadmat('ex5data1.mat', squeeze_me=True)

X = mat['X']
y = mat['y']
x = np.array(np.insert(np.matrix(X),[0],np.matrix(np.ones(len(X))),axis=0)).T
x = polyFeatures(x,8)
x, mu, sigma = normalizarCaracteristica(x)
y = np.array(y)
Xval = mat['Xval']
yval = mat['yval']
xval = np.array(np.insert(np.matrix(Xval),[0],np.matrix(np.ones(len(Xval))),axis=0)).T
xval = polyFeatures(xval,8)
xval, mu, sigma = normalizarCaracteristica(xval)
yval = np.array(yval)

initialTheta = np.ones((x.shape[1],1))
alpha = 1e-2
lamda = 0

trainingError,valError = learningCurve(x,y,xval,yval,initialTheta,alpha,lamda)
trainExamples = range(1,len(x))

train, = plt.plot(trainExamples, trainingError, "k", c="blue", label="Train")
val, = plt.plot(trainExamples,valError, 'k' ,c="darkGreen", label="Cross Validation")
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.title('Polynomial Regression Learning Curve (lambda = 0.0)')
plt.legend(handles=[train, val], loc='upper right')
#plt.savefig('plot1.2.png')
plt.show()