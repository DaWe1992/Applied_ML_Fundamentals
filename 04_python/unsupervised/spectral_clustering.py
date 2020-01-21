# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:04:06 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

from sklearn.datasets import make_moons


# -----------------------------------------------------------------------------
# Generate and visualize data
# -----------------------------------------------------------------------------

X, _ = make_moons(150, noise=0.07, random_state=21)
fig, ax = plt.subplots(figsize=(9,7))
ax.set_title("Data", fontsize=18, fontweight='demi')
ax.scatter(X[:, 0], X[:, 1], s=50, cmap="viridis")


# -----------------------------------------------------------------------------
# Class SpectralClustering
# -----------------------------------------------------------------------------

class SpectralClustering:
    """
    Class SpectralClustering
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    

    def fit(self, X):
        """
        Fits the spectral clustering model to the data.
        
        :param X:           data to  be clustered
        :return:            cluster assignments
        """
        self.X = X
        # compute adjacency matrix (symmetric)
        A = cdist(X, X)
        A[np.where(A > 0.4)] = 0
        # compute degree matrix (diagonal)
        D = np.diag(np.sum(A, axis=1))
        # compute laplacian matrix
        L = D - A

        # perform eigen-decomposition of laplacian matrix
        eigval, eigvec = np.linalg.eig(L)
        
        c_assign = eigvec[:, 1].copy()
        c_assign[c_assign < 0] = 0
        c_assign[c_assign > 0] = 1
        
        return c_assign


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
        
if __name__ == "__main__":
    # perform spectral clustering
    sc = SpectralClustering()
    c_assign = sc.fit(X)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title("Data after spectral clustering", fontsize=18, fontweight="demi")
    ax.scatter(X[:, 0], X[:, 1],c=c_assign ,s=50, cmap="viridis")
