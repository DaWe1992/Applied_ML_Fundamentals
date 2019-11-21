# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:29:47 2019

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def plot_multivariate_normal(mu, cov):
    """
    Plots a bi-variate normal distribution with
    given mu and sigma.
    
    :param mu:      mean vector
    :param cov:     covariance matrix
    """ 
    plt.figure(figsize=(20, 10))
    
    # create mesh grid and multivariate normal
    x = np.linspace(-10, 10, 1000)
    y = np.linspace(-10, 10, 1000)
    X, Y = np.meshgrid(x, y)
    
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    Z = multivariate_normal([mu[0], mu[1]], cov).pdf(pos)
    
    ax = plt.axes(projection="3d")
    ax.contour3D(X, Y, Z, 50, cmap="viridis")
    
    plt.show()
    
    
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":    
    mu = np.asarray([0, 0])
    
    # -------------------------------------------------------------------------
    cov = np.asarray([[1, 0], [0, 1]])
    print(cov)
    
    plot_multivariate_normal(mu, cov)
    
    # -------------------------------------------------------------------------
    cov = np.asarray([[5, 4], [4, 6]])
    print(cov)
    
    plot_multivariate_normal(mu, cov)
    
    # -------------------------------------------------------------------------
    cov = np.asarray([[5, -4], [-4, 6]])
    print(cov)
    
    plot_multivariate_normal(mu, cov)
    
    # -------------------------------------------------------------------------
    cov = np.asarray([[5, 0], [0, 1]])
    print(cov)
    
    plot_multivariate_normal(mu, cov)
    