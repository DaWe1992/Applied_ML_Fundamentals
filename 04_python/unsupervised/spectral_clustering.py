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

from scipy import spatial
from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans
from sklearn.datasets import make_moons


# -----------------------------------------------------------------------------
# Generate and visualize data
# -----------------------------------------------------------------------------

X, _ = make_moons(150, noise=0.07, random_state=21)
fig, ax = plt.subplots(figsize=(9,7))
ax.set_title("Data", fontsize=18, fontweight="demi")
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
    

    def fit(self, X, method="knn", k=5, eps=0.3):
        """
        Fits the spectral clustering model to the data.
        
        :param X:           data to  be clustered
        :param method:      method of graph construction
                                - knn
                                - eps
        :param k:           number of nearest neighbors for knn graph construction
                                (for method = "knn", ignored for others)
        :param eps:         distance threshold for epsilon neighborhood graph construction
                                (for method = "eps", ignored for others)
        :return:            cluster assignments
        """
        self.X = X
        
        if method == "knn":
            A = self.__knn_graph(k)
        else:
            A = self.__eps_nbh(eps)
        
        # compute degree matrix (diagonal)
        D = np.diag(np.sum(A, axis=1))
        # compute laplacian matrix
        L = D - A

        # perform eigen-decomposition of laplacian matrix
        eigval, eigvec = np.linalg.eig(L)
        
        U = eigvec
        # perform k-means on the spectral embeddings
        kmeans = KMeans(n_clusters=2, random_state=0).fit(U)
        
        return kmeans.labels_
    
    
    def __knn_graph(self, k):
        """
        Computes the knn graph.
        
        :param k:           number of nearest neighbors
        :return:            knn graph
        """
        A = np.zeros((self.X.shape[0], self.X.shape[0]))
        
        tree = spatial.KDTree(self.X)
        for i, x in enumerate(self.X):
            ds, idx = tree.query(x, k=k + 1)
            for d, j in zip(ds[1:], idx[1:]):
                A[i, j] = 1 / d
                A[j, i] = 1 / d
                
        return A
    
    
    def __eps_nbh(self, eps):
        """
        Computes the epsilon neighborhood graph.
        
        :param eps:         epsilon threshold
        :return:            epsilon neighborhood graph
        """
        A = cdist(X, X)
        A[np.where(A > eps)] = 0
        
        return A


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
        
if __name__ == "__main__":
    # perform spectral clustering
    sc = SpectralClustering()
    c_assign = sc.fit(X, method="knn")
    
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title("Data after spectral clustering", fontsize=18, fontweight="demi")
    ax.scatter(X[:, 0], X[:, 1],c=c_assign ,s=50, cmap="viridis")
