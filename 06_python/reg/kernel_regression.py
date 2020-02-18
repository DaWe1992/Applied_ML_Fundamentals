# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:23:16 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np
import numpy.linalg as linalg


# -----------------------------------------------------------------------------
# Class KernelRegression
# -----------------------------------------------------------------------------

class KernelRegression:
    """
    Class KernelRegression.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    
    
    def fit(self, X, y, lam=0.001, kernel="gaussian"):
        """
        Performs kernel ridge regression.
        
        :param X:               training data (features)
        :param y:               training data (labels)
        :param lam:             regularization term lambda
        :kernel:                kernel function to use
                                    - linear
                                    - polynomial
                                    - gaussian
        """
        self.X = X
        self.y = y
        self.lam = lam
        
        if kernel == "polynomial":
            self.kernel = self.__polynomial_kernel
        elif kernel == "gaussian":
            self.kernel = self.__gaussian_kernel
        else:
            kernel = self.__linear_kernel
    
    
    def predict(self, X_q):
        """
        Predicts the labels of unseen data.
        
        :param X_q:             unseen data features
        :return:                labels of unseen data
        """
        # calculate kernel matrix K
        K = self.__kernel_matrix(self.X, self.X) + self.lam * np.eye(self.X.shape[0])
        # calculate kernel vector K_s
        K_s = self.__kernel_matrix(self.X, X_q)
        
        return self.y.T @ linalg.inv(K) @ K_s
    
    
    def __kernel_matrix(self, X, Y):
        """
        Calculates the covariance matrix of X and Y.
        
        :param X:       data set 1
        :param Y:       data set 2
        :return:        kernel matrix
        """
        # number of examples
        n_x = X.shape[0]
        n_y = Y.shape[0]
        K = np.zeros((n_x, n_y))
        
        # calculate covariance-matrix
        for i in range(n_x):
            for j in range(n_y):
                K[i, j] = self.kernel(X[i], Y[j])
                
        return K

    
    # -------------------------------------------------------------------------
    # Kernels
    # -------------------------------------------------------------------------
    
    def __linear_kernel(self, x, y):
        """
        Linear kernel. Returns the dot product of x and y.
        
        :param x:       data point 1
        :param y:       data point 2
        :return:
        """
        return x @ y
    
    
    def __polynomial_kernel(self, x, y, p=3):
        """
        Polynomial kernel.
        
        :param x:       data point 1
        :param y:       data point 2
        :param p:       degree of the polynomial
        :return:
        """
        return (1 + x @ y)**p
    
    
    def __gaussian_kernel(self, x, y, sigma=0.5):
        """
        Gaussian (RBF = radial basis function) kernel.
        
        :param x:       data point 1
        :param y:       data point 2
        :param sigma:   standard deviation
        :return:
        """
        return np.exp(-linalg.norm(x - y)**2 / (2 * (sigma**2)))
    