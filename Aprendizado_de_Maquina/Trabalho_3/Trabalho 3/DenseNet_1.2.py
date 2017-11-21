# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 14:23:27 2017

@author: Rebecca
"""

from keras.layers.core import Dense
from keras.optimizers import SGD,RMSprop,Adam
from NNet import NNet
from importarVariavel import importarDados
from normalizarCaracteristica import normalizarCaracteristica
from confusion_matrix import confusion_matrix

#Importing data
filepath = r"C:\Users\Rebecca\Desktop\credtrain.txt"
X,Y = importarDados(filepath,['ESTC', 'NDEP', 'RENDA', 'TIPOR', 'VBEM', 'NPARC',
                              'VPARC', 'TEL', 'IDADE', 'RESMS', 'ENTRADA', 'CLASSE'])
filepath = r"C:\Users\Rebecca\Desktop\credtest.txt"
X_test,Y_test = importarDados(filepath,['ESTC', 'NDEP', 'RENDA', 'TIPOR', 'VBEM', 'NPARC',
                              'VPARC', 'TEL', 'IDADE', 'RESMS', 'ENTRADA', 'CLASSE'])

#Normalization of input data
X = normalizarCaracteristica(X)[0]
X_test = normalizarCaracteristica(X_test)[0]

#Set parameters
input_shape=(11,)
layers=[Dense(5,activation='sigmoid',input_shape=input_shape),
        Dense(1,activation='sigmoid')]
loss='mean_squared_error'
optimizer=SGD(lr=0.1) #Test
metrics=['accuracy']
epochs=250
batch_size=100
validation_data=(X_test, Y_test)
shuffle=True
plot=True

model,history = NNet(X,Y,layers=layers,loss=loss,optimizer=optimizer,
                     metrics=metrics,epochs=epochs,batch_size=batch_size, verbose=2,
                     validation_data=validation_data,shuffle=shuffle, plot=plot)
model.summary()

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Confusion matrix:\n', str(confusion_matrix(Y_test, model.predict(X_test).round())))