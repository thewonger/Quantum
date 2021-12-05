# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:46:46 2020

@author: Jorge
"""

from tensorflow import keras
import numpy as np
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs= np.array([-1.0,0.0,1.0,2.0,3.0,4.0], dtype =float)
ys = np.array([-3.0,-1.0,1.0,3.0,5.0,7.0],dtype=float)