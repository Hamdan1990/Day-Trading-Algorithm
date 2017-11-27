# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:53:29 2017

@author: Hamda
"""

import tensorflow as tf


    

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


class RecurrentNeuralNetwork():
    
regressor = Sequential()

regressor.add(LSTM(units = 100, return_sequences = True, input_shape =( x_train.shape[1],3)))
regressor.add(Dropout(0.2))
# units is the number of neurons in the layer, since predicting the stock price is complex hence we choose high dimensionality -
# - which means having many layers and each layer having high number of neurons

# you put return_sequences = true if there is an additional LSTM layer in front

# For input_shape , we just need to specify the x_shape along the of timestep and indicators as the actual values of x_train - 
# - has already been taken into account by the LSTM constructor

regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 100))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer='adam', loss= 'mean_squared_error')