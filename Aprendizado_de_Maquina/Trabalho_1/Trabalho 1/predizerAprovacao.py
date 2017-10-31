# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 18:17:51 2017

@author: Rebecca
"""

import numpy as np
from gdreglog import gd
from funcaoCustoRegressaoLogistica import htheta
from importarVariavel import importarDados

filepath = "\ex2data1.txt"
x,y = importarDados(filepath,['Exam 1', 'Exam 2', 'Admitted'])

theta = np.zeros(3)
alpha = 0.01

theta,Jtheta,converged = gd(x,y,theta,alpha)


p_test = htheta(np.matrix(np.array([1.0,45.,85.])),np.matrix(theta))
print 'Probabilidade de aprovacao para notas 45 e 85 = {0}%'.format(np.float(p_test))

def predizer(example, theta):
    example = np.matrix(example)
    theta = np.matrix(theta)
    probability = htheta(example,theta)
    return [1 if p >= 0.5 else 0 for p in probability]

def acertos(x, theta):
    predictions = predizer(x, theta)  
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]  
    accuracy = (sum(map(int, correct)) % len(correct))
    return accuracy
    
print 'Porcentagem de acertos = {0}%'.format(acertos(x, theta))