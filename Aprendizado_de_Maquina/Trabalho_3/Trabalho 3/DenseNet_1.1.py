# -*- coding. utf-8 -*-
"""
Created on Mon Nov 13 14.48.52 2017

@author: Rebecca
"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt

P = np.array(np.matrix([[0.4046,0.3786,0.7010,0.8608,0.5947],
						[0.9974,0.8479,0.6201,0.4031,0.9653],
						[0.3764,0.9214,0.9331,0.7514,0.6914],
						[0.6043,0.3494,0.1438,0.6035,0.4111]]).T )
T = np.array([[0],[1],[0],[1],[1]])

model = Sequential()
l1 = Dense(3, input_dim=4, activation='sigmoid')
model.add(l1)
l2 = Dense(1, activation='sigmoid')
model.add(l2)
sgd = SGD(lr=0.1)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
history = model.fit(P, T, batch_size=1, epochs=10000, verbose=0)

# summarize history for accuracy
plt.plot(history.history['acc'], label="Training")
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='bottom left')
plt.show()
        
# summarize history for loss
plt.plot(history.history['loss'], label="Training")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()

model.summary()

print ('Prediction of P',model.predict(P).round())

score = model.evaluate(P, T, verbose=0)
print('Training loss:', score[0])
print('Training accuracy:', score[1])