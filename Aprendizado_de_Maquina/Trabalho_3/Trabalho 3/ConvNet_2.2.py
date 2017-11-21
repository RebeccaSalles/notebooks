# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:05:47 2017

@author: Rebecca
"""

import numpy as np
import keras
from keras.layers.core import Dense,Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D
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
print ("y = "+str(Y[0,index])+". It's a "+classes[Y[0,index]].decode("utf-8")+" picture.")

# Exploring the dataset 
print ("Number of training examples: " + str(X.shape[0]))
print ("Number of testing examples: " + str(X_test.shape[0]))
print ("Each image is of size: (" + str(X.shape[1]) + ", " + str(X.shape[1]) + ", 3)")

#Pre-process data
# Standardize data to have feature values between 0 and 1.
X_std = (X/255).astype('float32')
X_test_std = (X_test/255).astype('float32')
print ("X's shape: " + str(X_std.shape))
print ("X_test's shape: " + str(X_test_std.shape))
# convert class vectors to binary class matrices
num_classes = len(classes)
Y_categ = keras.utils.to_categorical(Y, num_classes)
Y_test_categ = keras.utils.to_categorical(Y_test, num_classes)

#Set parameters
input_shape = (X_std.shape[1], X_std.shape[2], X_std.shape[3])
layers=[Conv2D(16, kernel_size=(3, 3),activation='relu',input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')]
loss='categorical_crossentropy'
optimizer=RMSprop(lr=0.0001, decay=1e-5)
metrics=['accuracy']
epochs=12
batch_size=64
validation_data=(X_test_std, Y_test_categ)
shuffle=True
plot=True

model,history = NNet(X_std,Y_categ,layers=layers,loss=loss,optimizer=optimizer,
                     metrics=metrics,epochs=epochs,batch_size=batch_size, verbose=1,
                     validation_data=validation_data,shuffle=shuffle, plot=plot)
model.summary()

score = model.evaluate(X_test_std, Y_test_categ, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Confusion matrix:\n', str(confusion_matrix(np.argmax(Y_test_categ,axis=1), np.argmax(model.predict(X_test_std).round(),axis=1))))