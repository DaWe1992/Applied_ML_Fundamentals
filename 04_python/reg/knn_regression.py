# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 08:51:00 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np

from scipy import spatial


# -----------------------------------------------------------------------------
# Class KnnRegression
# -----------------------------------------------------------------------------

class KnnRegression:
    """
    Class KnnRegression.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    
    
    def fit(self, X, y, k=5):
        """
        Fits a knn regression model to the data.
        
        :param X:           training data (features)
        :param y:           training data (labels)
        :param k:           number of neighbors to consider
        """
        self.X = X
        self.y = y
        self.k = k
        
        
    def predict(self, X):
        """
        Predicts the label of unseen data.
        
        :param X:           unseen data
        :return:            labels of unseen data
        """
        y_pred = []
        
        # create kd-tree
        tree = spatial.KDTree(self.X)
        # go over all unseen data instances
        for i, x in enumerate(X):
            ds, idx = tree.query(np.asarray([x]), k=self.k)
            # compute mean of their nearest neighbors
            y_pred.append(np.mean(self.y[idx]))
            
        return np.asarray(y_pred)
        