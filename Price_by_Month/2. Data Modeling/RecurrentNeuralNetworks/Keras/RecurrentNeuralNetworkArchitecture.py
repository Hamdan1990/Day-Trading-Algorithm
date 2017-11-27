# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:53:29 2017

@author: Hamdan
"""

import tensorflow as tf


    

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt

class RecurrentNeuralNetwork():
    
    def __init__(self, units_per_layer = 50, drop_out_rate = 0.5):
        self.units = units_per_layer
        self.drop = drop_out_rate
        
        
    def CreateRegressorRNN(self,optimizer, loss, input_shape ):
        
        regressor = Sequential()

        regressor.add(LSTM(units = self.units, return_sequences = True, input_shape =input_shape))
        regressor.add(Dropout(self.drop))
        
        regressor.add(LSTM(units = self.units, return_sequences = True))
        regressor.add(Dropout(self.drop))
        
        regressor.add(LSTM(units = self.units, return_sequences = True))
        regressor.add(Dropout(self.drop))
        
        regressor.add(LSTM(units = self.units, return_sequences = True))
        regressor.add(Dropout(self.drop))
        
        regressor.add(LSTM(units = self.units, return_sequences = True))
        regressor.add(Dropout(self.drop))
        
        regressor.add(LSTM(units = self.units))
        regressor.add(Dropout(self.drop))
        
        regressor.add(Dense(units = 1))
        regressor.compile(optimizer=optimizer, loss= loss)
        
        self.regressor = regressor
        
    def Fit(self, x,y,epoch = 10, batch_size = 32):
        
        history  = self.regressor.fit(x, y, epochs=epoch, batch_size=batch_size)
        
        # summarize history for loss
        plt.figure(figsize = (16,10))
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('Epoch_Loss.png')
        plt.show()
        
    def Predict(self,test):
        prediction = self.regressor.predict(test)
        return prediction
    
    def SaveModel(self):
        # serialize model to JSON
        model_json = self.regressor.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.regressor.save_weights("model.h5")
        print("Saved model to disk")
        
    
 