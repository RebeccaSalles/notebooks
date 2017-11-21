# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 11:41:51 2017

@author: Rebecca
"""

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt

def NNet(P,T,layers=[Dense(3,activation='sigmoid',input_shape=(4,)),
                     Dense(1,activation='sigmoid')],
         loss='mean_squared_error',optimizer=SGD(lr=0.1),metrics=['binary_accuracy'],
         epochs=100000,batch_size=1, verbose=0,validation_split=0.0,
         validation_data=None, shuffle=True, plot=True):
    
    model = Sequential()
    for layer in layers:
        model.add(layer)
        
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = model.fit(P, T, batch_size=batch_size, epochs=epochs, verbose=verbose,
                        validation_split=validation_split, validation_data=validation_data,
                        shuffle=shuffle)
    
    if(plot):
        if 'acc' in history.history.keys():
            # summarize history for accuracy
            plt.plot(history.history['acc'], label="Training")
            if validation_data != None:
                label="Test"
                plt.plot(history.history['val_acc'], label=label)
            elif validation_split > 0.0:
                label="Validation"
                plt.plot(history.history['val_acc'], label=label)        
            plt.title('Model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(loc='upper left')
            plt.show()
        
        if 'loss' in history.history.keys():
            # summarize history for loss
            plt.plot(history.history['loss'], label="Training")
            if validation_data != None:
                label="Test"
                plt.plot(history.history['val_loss'], label=label)
            elif validation_split > 0.0:
                label="Validation"
                plt.plot(history.history['val_loss'], label=label)        
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(loc='upper right')
            plt.show()
    
    return model,history