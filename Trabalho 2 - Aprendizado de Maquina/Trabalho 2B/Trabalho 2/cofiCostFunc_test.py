# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:20:50 2017

@author: Rebecca
"""

import numpy as np
from cofiCostFunc import cofiCostFunc
import scipy.io as spio

mat = spio.loadmat('ex8_movies.mat', squeeze_me=True)

Y = mat['Y']
R = mat['R']

users = 4  
movies = 5  
features = 3

params_data = spio.loadmat('ex8_movieParams.mat', squeeze_me=True)  
X = params_data['X']  
Theta = params_data['Theta']

X_sub = X[:movies, :features]  
Theta_sub = Theta[:users, :features]  
Y_sub = Y[:movies, :users]  
R_sub = R[:movies, :users]

params = np.concatenate((np.ravel(X_sub), np.ravel(Theta_sub)))

J, grad = cofiCostFunc(params, Y_sub, R_sub, features)  
print "Custo: " + str(J) + "\nGradiente: " + str(grad)