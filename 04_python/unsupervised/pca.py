# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:18:08 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np


# -----------------------------------------------------------------------------
# Class PCA
# -----------------------------------------------------------------------------

class PCA():
    """
    Class PCA.
    Implements Principal Component Analysis.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    
    
    def fit_transform(self, X, n_components):
        """
        Reduces the dimensionality of the data.
        
        :param X:               high dimensional data
        :param n_components:    number of components
        """
        # compute scatter / covariance matrix
        sigma = np.cov(X, rowvar=False)

        # compute Eigenvectors and Eigenvalues and visualize
        eig_val, eig_vec = np.linalg.eig(sigma)

        # sort eigenvectors by decreasing eigenvalues
        idx = eig_val.argsort()[::-1]   
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:,idx]
        
        U = eig_vec[:, :n_components]
        
        # transform the samples onto the new subspace
        return X @ U
    