# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 14:40:57 2017

@author: Rebecca
"""

import numpy as np
import pandas as pd  
#import os

#filepath = data columns file path
#names = given column names
def importarDados(filepath,names,sep='\t'):
    #path = os.getcwd() + filepath  
    data = pd.read_csv(filepath, header=None, sep=sep, names=names)

    # separa os conjuntos de dados x (caracteristicas) e y (alvo)
    cols = data.shape[1]
    x = data.iloc[:,0:cols-1]
    y = data.iloc[:,cols-1:cols]
    
    # converte os valores em numpy arrays
    x = np.array(x.values)  
    y = np.array(y.values)
    
    return x,y