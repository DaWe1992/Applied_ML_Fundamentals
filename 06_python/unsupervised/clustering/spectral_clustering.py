# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:04:06 2020
Cf. https://github.com/pin3da/spectral-clustering

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np
import scipy

from sklearn.cluster import KMeans


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

    
    def fit(self, X, n_clusters):
        """
        Fits a spectral clustering model to the data.
        
        :return:                cluster assignments
        """
        self.X = X
        self.n = X.shape[0]
        self.n_clusters = n_clusters
        
        # compute Laplacian matrix
        L = self.__laplacian(self.__compute_affinity())
        # cluster the data points
        c_assign = self.__cluster(L)
        
        return c_assign
    
    
    def __cluster(self, L):
        """
        Clusters the data points.
        
        :param L:               Laplacian matrix
        """
        # perform eigen-decomposition of L (only 'n_clusters' elements)
        eig_val, eig_vect = scipy.sparse.linalg.eigs(L, self.n_clusters)
        U = eig_vect.real
        # normalize eigenvectors and perform k-Means clustering
        rows_norm = np.linalg.norm(U, axis=1, ord=2)
        c_assign = self.__k_means((U.T / rows_norm).T)
        
        return c_assign
    
    
    def __laplacian(self, A):
        """
        Computes the symetric normalized laplacian.
        
        :param A:               affinity matrix
        :return:                normalized graph Laplacian matrix
        """
        D = np.zeros(A.shape)
        w = np.sum(A, axis=0)
        D.flat[::len(w) + 1] = w ** (-0.5)  # set the diagonal of D to w
        
        return D.dot(A).dot(D)
    
    
    def __k_means(self, U):
        """
        Applies k-Means clustering.
        
        :param U:               transformed data
        :return:                cluster assignments
        """
        kmeans = KMeans(n_clusters=self.n_clusters)
        
        return kmeans.fit(U).labels_
    
    
    def __compute_affinity(self):
        """
        Computes the affinity matrix for the data.
        
        :return:                affinity matrix
        """
        A = np.zeros((self.n, self.n))
        sig = []
        
        # compute pairwise distances
        for i in range(self.n):
            dists = []
            for j in range(self.n):
                dists.append(np.linalg.norm(self.X[i] - self.X[j]))
                
            dists.sort()
            sig.append(np.mean(dists[:5]))
    
        # compute squared exponential kernel
        for i in range(self.n):
            for j in range(self.n):
                A[i][j] = self.__squared_exponential(self.X[i], self.X[j], sig[i], sig[j])
                
        return A
    
    
    def __squared_exponential(self, x, y, sig1=0.8, sig2=1):
        """
        Squared exponential kernel.
        
        :param x:               data point 1
        :param y:               data point 2
        :sig1:                  sigma 1
        :sig2:                  sigma 2
        :return:                squared exponential value
        """
        return np.exp(-np.linalg.norm(x - y)**2 / (2 * sig1 * sig2))