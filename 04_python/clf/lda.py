# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 07:50:12 2019

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np


# -----------------------------------------------------------------------------
# Class LDA
# -----------------------------------------------------------------------------

class LDA():
    """
    LDA class.
    Implements Fisher's Linear Discriminant.
    """
    
    def __init__(self, n_dims=1):
        """
        Constructor.
        
        :param num_dims:        number of dimensions to project onto
        """
        self.n_dims = n_dims

    
    def fit(self, X, y):
        """
        Fits the LDA model.
        
        :param X:               data features (training data)
        :param y:               data labels (training data)
        """
        self.n = X.shape[0]
        self.m = X.shape[1]
        
        self.class_data = {}
        self.classes = np.unique(y)
        
        # get data for each class
        for c in self.classes:
            self.class_data[c] = X[np.where(y == c)[0],:]
        
        # ---------------------------------------------------------------------
        # calculate means
        # ---------------------------------------------------------------------
        # calculate means for each class
        self.class_means = {}
        for c in self.classes:
            self.class_means[c] = np.mean(self.class_data[c], axis=0)
            
        # calculate the overall mean of all the data
        self.mean = np.mean(X, axis=0)

        # ---------------------------------------------------------------------
        # calculate covariances
        # ---------------------------------------------------------------------
        # calculate between-class covariance matrix
        S_B = self.__variance_between()
        
        # calculate within-class covariance matrix
        S_W = self.__variance_within()

        # ---------------------------------------------------------------------
        # find eigenvectors with largest eigenvalues
        # ---------------------------------------------------------------------
        # find <eigenvalue, eigenvector> pairs for dot(inv(S_W), S_B)
        mat = np.linalg.pinv(S_W) @ S_B
        eigvals, eigvecs = np.linalg.eig(mat)
        eiglist = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]

        # sort the eigvals in decreasing order
        eiglist = sorted(eiglist, key=lambda x : x[0], reverse=True)

        # take the first 'num_dims' eigvectors
        self.w = np.array([eiglist[i][1] for i in range(self.n_dims)])
        # calculate bias
        self.w0 = -np.dot(self.w, self.mean)[0]
        
        
    def predict(self, X):
        """
        Predicts the labels of unseen data.
        
        :param X:               data features (test data)
        :return:                labels for test data instances
        """
        # project data
        return np.sign(X @ self.w.T + self.w0)
    
    
    def __variance_within(self):
        """
        Computes the within-class variance.
        
        :return:                within class covariance matrix
        """
        S_W = np.zeros((self.m, self.m)) 
        for c in self.classes: 
            tmp = np.subtract(
                self.class_data[c].T,
                np.expand_dims(self.class_means[c], axis=1)
            )
            S_W += tmp @ tmp.T
            
        return S_W
    
    
    def __variance_between(self):
        """
        Computes the between-class variance.
        
        :return:                between class covariance matrix
        """
        S_B = np.zeros((self.m, self.m))
        
        for c in self.classes:
            S_B += np.outer(
                (self.class_means[c] - self.mean), 
                (self.class_means[c] - self.mean)
            )
                
            return S_B
    