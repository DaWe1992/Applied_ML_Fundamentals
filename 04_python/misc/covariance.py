# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:12:53 2019

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Covariance
# -----------------------------------------------------------------------------

def cov(X):
    """
    Calculates covariance matrix.
    
    :param X:       data
    :return:        covariance matrix
    """
    # initialize covariance matrix
    cov = np.zeros((2, 2))
    # compute mean per dimension
    x_m = np.mean(X, axis=0)
    
    # compute outer products
    for i in range(X.shape[0]):
        cov += np.outer((X[i] - x_m), (X[i] - x_m))
    cov /= X.shape[0]
    
    return cov


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
    
if __name__ == "__main__":
    x = np.random.normal(0, 1, 500)
    y = np.random.normal(0, 1, 500)
    X = np.vstack((x, y)).T

    plt.scatter(X[:, 0], X[:, 1])
    plt.title("Generated Data")
    plt.show()

    print(cov(X))
