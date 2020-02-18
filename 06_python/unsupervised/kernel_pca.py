# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:18:32 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np

from scipy import exp
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform


# -----------------------------------------------------------------------------
# Class KernelPCA
# -----------------------------------------------------------------------------

class KernelPCA:
    """
    Class KernelPCA.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    
    
    def fit_transform(self, X, n_components, gamma=0.5):
        """
        Implements RBF kernel PCA.
        
        :param X:               (n x m)-dimensional data set
        :n_components:          number of components
        :param gamma:           rbf kernel parameter
        :return:                data set reduced in dimensionality
        """
        # calculate pairwise squared Euclidean distances
        sq_dists = pdist(X, "sqeuclidean")
    
        # convert the pairwise distances into a symmetric matrix
        mat_sq_dists = squareform(sq_dists)
    
        # compute the kernel-matrix.
        K = exp(-gamma * mat_sq_dists)
    
        # center the symmetric kernel-matrix
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
        # obtain eigenvalues in descending order with corresponding
        # eigenvectors from the symmetric matrix
        eigvals, eigvecs = eigh(K)
    
        # obtain the i eigenvectors that correspond to the i highest eigenvalues
        X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))
    
        return X_pc
    