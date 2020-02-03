# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 08:42:29 2020

@author: Daniel Wehner
Implementation of an AutoEncoder.
Unlike PCA, an AutoEncoder is able to project the data
onto a non-linear subspace.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import torch
import torch.nn.functional as F

from torch import nn, optim


# -----------------------------------------------------------------------------
# Class Net
# -----------------------------------------------------------------------------

class Net(nn.Module):
    """
    Class Net.
    """
        
    def __init__(self, dims):
        """
        Constructor.
        
        :param dims:        layer dimensionalities
        """
        super().__init__()
        self.dims = dims
        self.__model_fn()
    
    
    def __model_fn(self):
        """
        Specifies the network.
        """
        layers = []
        for i in range(len(self.dims) - 1):
            layers.append(
                torch.nn.Linear(self.dims[i], self.dims[i + 1]))
            
        self.layers = nn.ModuleList(layers)
                

    def forward(self, X):
        """
        Performs a forward-pass on the data.
        
        :param X:           network input
        :return:            network output
        """
        for layer in self.layers:
            X = F.relu(layer(X))
            
        return X


# -----------------------------------------------------------------------------
# Class AutoEncoder
# -----------------------------------------------------------------------------

class AutoEncoder(nn.Module):
    """
    Class AutoEncoder.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        super().__init__()
            
    
    def fit(self, X, n_components=2, denoising=False):
        """
        Fits the auto-encoder to the data.
        
        :param X:               training data
        :param n_components:    number of components (reduced dimensionality)
        :param denoising:       flag indicating whether to add noise
                                    to the input (= denoising auto-encoder)
        """
        # encoder network
        dims = [X.shape[1], 16, 8, n_components]
        self.enc = Net(dims=dims)
        # decoder network
        dims.reverse()
        self.dec = Net(dims=dims)
        
        self.X_out = torch.FloatTensor(X)
        self.X_in = torch.FloatTensor(X)
        # add random noise
        if denoising: self.X_in += torch.randn(X.shape)
        
        # jointly train the networks
        self.__train()
        
    
    def __train(self):
        """
        Trains the auto-encoder.
        """
        # loss and optimizer
        criterion = torch.nn.MSELoss()
        e_optimizer = optim.Adam(self.enc.parameters(), lr=0.001)
        d_optimizer = optim.Adam(self.dec.parameters(), lr=0.001)

        # perform training
        # ---------------------------------------------------------------------
        self.train()
        n_epochs = 5000
        for epoch in range(n_epochs):
            # IMPORTANT !!!
            # set the gradients back to zero
            d_optimizer.zero_grad()
            e_optimizer.zero_grad()
            # forward pass / get activations for last layer
            X_pred = self(self.X_in)
            # compute loss
            loss = criterion(X_pred, self.X_out)
           
            print("Epoch {}: train loss: {}".format(epoch, loss.item()))
            # backward pass
            loss.backward()
            d_optimizer.step()
            e_optimizer.step()
            
            
    def transform(self):
        """
        Transforms the data from high-dimensional to low-dimensional.
        
        :return:            low-dimensional representation
        """
        return self.enc(self.X_in).detach().numpy()
        
        
    def forward(self, X):
        """
        Performs forward pass.
        
        :param x:           network input
        :return:            network output
        """
        return self.dec(self.enc(X))
    