#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:21:36 2019

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# scipy and numpy
# -----------------------------------------------------------------------------
import numpy as np

from scipy.spatial.distance import cdist


# -----------------------------------------------------------------------------
# Class kNN
# -----------------------------------------------------------------------------

class kNN():
    """
    Class kNN (k-nearest neighbors)
    """
    
    def __init__(self, n_neighbors):
        """
        Constructor.
        
        :param n_neighbors:     number of neighbors to be considered (= k)
        """
        self.n_neighbors = n_neighbors
        
        
    def fit(self, X, y):
        """
        Fits a kNN model to the data.
        
        :param X:               data features (training data)
        :param y:               data labels (training data)
        """
        self.X = X.reshape(-1, X.shape[1])
        self.y = y
        
    
    def predict(self, X):
        """
        Predicts the labels for the unseen instances.
        
        :param x:               unseen instances, features
        :return:                labels for the unseen instances
        """
        n_instances = X.shape[0]
        pred = np.zeros(n_instances)
        
        for i in range(n_instances):
            pred[i] = self.__predict_single(X[i, :])
            
        return pred
    
    
    def __predict_single(self, x):
        """
        Predicts the label for the unseen instance.
        
        :param x:               unseen instance, features
        :return:                label for the unseen instance
        """
        # distances of unseen instances to data points
        sq_dist = cdist(x.reshape(-1, self.X.shape[1]), self.X)**2
        # calculate the nearest neighbors
        nbs = self.y[sq_dist.argsort()[0]][0:self.n_neighbors]
        
        # get label with maximum count
        return np.argmax(np.bincount(nbs))
    