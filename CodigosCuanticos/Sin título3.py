# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:48:06 2020

@author: Jorge
"""


from tensorflow import keras
import numpy as np


model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')


#y=2x -1

xs= np.array([-1.0,0.0,1.0,2.0,3.0,4.0], dtype =float)
ys = np.array([-3.0,-1.0,1.0,3.0,5.0,7.0],dtype=float)

model.fit(xs,ys,epochs=1000)
print(model.predict([10]))


#le damos x=10 y nos devuelve 19

