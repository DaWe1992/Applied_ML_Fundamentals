# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:26:11 2020
IRLS - Iterative Reweighted Least Squares (logistic regression with Newton-Raphson method)
cf. Bishop et al. "Pattern Recognition and Machine Learning", page 207.

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np


# -----------------------------------------------------------------------------
# Class IRLS
# -----------------------------------------------------------------------------

class IRLS:
    """
    Class IRLS.
    """
    
    def __init__(self, poly=True):
        """
        Constructor.
        
        :param poly:            usage of polynomial features
        """
        self.poly = poly
    
    
    def fit(self, X, y, n_iter=10):
        """
        Fits the model to the data.
        
        :param X:               data features (training data)
        :param y:               data labels (training data)
        :param n_iter:          maximum number of iterations
        """
        Phi = X
        if self.poly:
            Phi = self.__poly_features(X)
        Phi_hat = self.__concat_ones(Phi)
        
        self.theta = np.asarray([0.00] * (Phi_hat.shape[1]))
        
        # perform training iterations
        for i in range(n_iter):
            # compute predictions
            pred = self.__compute_class_prob(Phi_hat)
            # compute diagonal matrix of predictions
            R = np.diag(pred * (1 - pred))
            # update theta
            self.theta -= np.linalg.inv(Phi_hat.T @ R @ Phi_hat) @ Phi_hat.T @ (pred - y) 
    
    
    def predict(self, X, probs=False):
        """
        Predicts the labels of unseen data.
        
        :param X:               data features (test data)
        :param probs:           flag indicating if class labels or probabilities
                                should be returned
        :return:                labels of unseen data
        """
        Phi = X
        # create polynomial features
        if self.poly:
            Phi = self.__poly_features(X)
        # concatenate ones column
        Phi_hat = self.__concat_ones(Phi)
        # compute probabilities for class one
        p = self.__compute_class_prob(Phi_hat)
        
        return p if probs else np.round(p)
    
    
    def __compute_class_prob(self, Phi_hat):
        """
        Computes the probability for class one.
        
        :param Phi_hat:         data features
        :return:                probabilities for class one
        """
        return 1 / (1 + np.exp(np.negative(Phi_hat @ self.theta)))
    
    
    def __concat_ones(self, X):
        """
        Concatenates a column of ones for the bias.
        
        :param X:               raw features
        :return:                features with ones column
        """
        return np.concatenate(
            (np.ones((X.shape[0], 1)), X.reshape(-1, X.shape[1])),
            axis=1        
        )
        
        
    def __poly_features(self, X):
        """
        Computes polynomial features.
        
        :param X:               data points
        :return:                polynomial features
        """
        X1 = X[:, 0]
        X2 = X[:, 1]
        
        return np.asarray([
            X1, X2,                                 # degree 1
            X1*X1, X1*X2, X2*X2,                    # degree 2
            X1**3, 3*X1*X1*X2, 3*X1*X2*X2, X2**3    # degree 3
        ]).T
    