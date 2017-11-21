# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:05:47 2017

@author: Rebecca
"""

import numpy as np
import keras
from keras.layers.core import Dense
from keras.optimizers import RMSprop
from NNet import NNet
from confusion_matrix import confusion_matrix
from load_dataset_h5 import load_dataset
import matplotlib.pyplot as plt

#Importing data
X,Y,X_test,Y_test,classes = load_dataset()

# Example of a picture
index = 10
plt.imshow(X[index])
plt.show()
print ("y = "+str(Y[0,index])+". It's a "+classes[Y[0,index]].decode("utf-8")+" picture.")

# Exploring the dataset 
print ("Number of training examples: " + str(X.shape[0]))
print ("Number of testing examples: " + str(X_test.shape[0]))
print ("Each image is of size: (" + str(X.shape[1]) + ", " + str(X.shape[1]) + ", 3)")

#Pre-process data
# Reshape the training and test examples 
X_flatten = X.reshape(X.shape[0], -1).astype('float32')   # The "-1" makes reshape flatten the remaining dimensions
X_test_flatten = X_test.reshape(X_test.shape[0], -1).astype('float32')
# Standardize data to have feature values between 0 and 1.
X_flatten = X_flatten/255
X_test_flatten = X_test_flatten/255
print ("train_x's shape: " + str(X_flatten.shape))
print ("test_x's shape: " + str(X_test_flatten.shape))
# convert class vectors to binary class matrices
num_classes = len(classes)
Y_categ = keras.utils.to_categorical(Y, num_classes)
Y_test_categ = keras.utils.to_categorical(Y_test, num_classes)

#Set parameters
input_shape=(12288,)
layers=[Dense(512,activation='relu',input_shape=input_shape),
        Dense(num_classes,activation='softmax')]
loss='categorical_crossentropy'
optimizer=RMSprop(lr=0.0001) #Test
metrics=['accuracy']
epochs=25
batch_size=128
validation_data=(X_test_flatten, Y_test_categ)
shuffle=True
plot=True

model,history = NNet(X_flatten,Y_categ,layers=layers,loss=loss,optimizer=optimizer,
                     metrics=metrics,epochs=epochs,batch_size=batch_size, verbose=1,
                     validation_data=validation_data,shuffle=shuffle, plot=plot)
model.summary()

score = model.evaluate(X_test_flatten, Y_test_categ, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Confusion matrix:\n', str(confusion_matrix(np.argmax(Y_test_categ,axis=1), np.argmax(model.predict(X_test_flatten).round(),axis=1))))