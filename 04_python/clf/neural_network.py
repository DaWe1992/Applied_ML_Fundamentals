# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 17:37:56 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np


# -----------------------------------------------------------------------------
# Class NeuralNetwork
# -----------------------------------------------------------------------------

class NeuralNetwork:
    """
    Class NeuralNetwork.
    """
    
    def __init__(self, layers):
        """
        Constructor.
        
        :param layers:  specification of hidden layer sizes, e.g. [30, 10]
        """
        self.layers = layers
    
    
    def fit(self, X, y, batch_size=16):
        """
        Fits a neural network model to the data.
        
        :param X:       data features (training data)
        :param y:       data labels (training data)
        """
        self.X = X
        self.y = y
        # append number of neurons for last layer (data-dependent)
        self.layers.append(np.unique(self.y).shape[0])
        
        # layer initializations
        self.params = list()
        self.params.append(self.__init_layer(self.layers[0], X.shape[1]))
        for l in range(1, len(self.layers)):
            self.params.append(
                self.__init_layer(self.layers[l], self.layers[l - 1]))
        
        p = self.X
        for l in range(len(self.layers)):
            p = self.__act_func(p @ self.params[l].T)
    
    
    def predict(self, X):
        """
        Predicts the labels of unseen data.
        
        :param X:               data features (test data)
        :return:                labels for test data instances
        """
        pass
    
    
    def __forward(self, X):
        """
        Performs a forward-pass.
        
        :param X:       data
        """
        pass
    
    
    def __backward(self):
        """
        Performs a backward-pass
        """
        pass
    
    
    def __init_layer(self, s0, s1):
        """
        Initializes a layer.
        """
        return np.random.uniform(-1, 1, (s0, s1))
    
    
    def __act_func(self, X):
        """
        Sigmoid activation function.
        
        :param X:       data
        :return:        sigmoid activations
        """
        return 1 / (1 + np.exp(np.negative(X)))
    
    
    def __next_batch(self, n=1):
        """
        Returns a batch of size n.
        
        :param n:               size of the batch
        :return:                batch of size n
        """
        idx = np.arange(0 , self.n)
        np.random.shuffle(idx)
        idx = idx[:n]
        X_shuffle = [self.X[i] for i in idx]
        y_shuffle = [self.y[i] for i in idx]
    
        return np.asarray(X_shuffle), np.asarray(y_shuffle)
    