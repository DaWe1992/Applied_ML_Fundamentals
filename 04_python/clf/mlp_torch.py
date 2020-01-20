# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:33:25 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import torch
import numpy as np


# -----------------------------------------------------------------------------
# Class MLP
# -----------------------------------------------------------------------------

class MLP(torch.nn.Module):
    """
    Class MLP.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        super(MLP, self).__init__()
        
        
    def __model_fn(self, n_input, n_out, n_hidden_1=64, n_hidden_2=64):
        """
        Specifies the network.
        
        :param n_input:     number of input dimensions
        :param n_out:       number of output units (classes)
        :param n_hidden_1:  number of hidden units in first hidden layer
        :param n_hidden_2:  number of hidden units in second hidden layer
        """
        self.fc1 = torch.nn.Linear(n_input, n_hidden_1)
        self.act1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(n_hidden_1, n_hidden_2)
        self.act2 = torch.nn.ReLU()
        self.out = torch.nn.Linear(n_hidden_2, n_out)
        self.softmax = torch.nn.Softmax(-1)
        
        
    def forward(self, X):
        """
        Performs a forward pass on the data.
        
        :param X:           data to be passed through the network
        :return:            last layer activations
        """
        X = self.act1(self.fc1(X))
        X = self.act2(self.fc2(X))
        X = self.softmax(self.out(X))
        
        return X
    
    
    def fit(self, X, y):
        """
        Fits the model to the data.
        
        :param X:           training data (features)
        :param y:           training data (labels)
        """
        self.X = torch.FloatTensor(X)
        
        n_classes = np.unique(y).shape[0]
        # one-hot encoding
        self.y = torch.FloatTensor(X.shape[0], n_classes)
        self.y.zero_()
        self.y[torch.arange(X.shape[0]), y] = 1
        
        # create model
        self.__model_fn(X.shape[1], n_classes)
        
        # loss and optimizer
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

        # perform training
        # ---------------------------------------------------------------------
        self.train()
        n_epochs = 1000
        for epoch in range(n_epochs):
            # IMPORTANT !!!
            # set the gradients back to zero
            optimizer.zero_grad()
            # forward pass / get activations for last layer
            y_pred = self(self.X)
            # compute loss
            loss = criterion(y_pred.squeeze(), self.y)
           
            print("Epoch {}: train loss: {}".format(epoch, loss.item()))
            # backward pass
            loss.backward()
            optimizer.step()
            
            
    def predict(self, X):
        """
        Predicts the label of unseen data instances.
        
        :param X:           unseen data (features)
        :return:            labels of unseen data
        """
        act = self(torch.FloatTensor(X))
        _, ind = torch.max(act, 1)
        
        return ind
    