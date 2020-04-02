# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:45:52 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import itertools
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram


# -----------------------------------------------------------------------------
# Class AgglomerativeClustering
# -----------------------------------------------------------------------------

class AgglomerativeClustering():
    """
    Class AgglomerativeClustering.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    
    
    def fit(self, X, n_cluster, method="complete_link", dendrogram=True):
        """
        Fits an agglomerative clustering model to the data.
        
        :param X:               training data
        :param n_cluster:       number of clusters
        :param method:          clustering method, e.g.:
                                    - single_link
                                    - complete_link
        :param dendrogram:      flag indicating whether to plot the dendrogram
        :return:                cluster assignments
        """
        self.X = X
        self.n = X.shape[0]
        self.n_cluster = n_cluster
        self.method = method
        
        # cluster the data
        c_assign, linkage_mat = self.__cluster()
        
        # plot dendrogram if specified
        if dendrogram:
            self.__dendrogram(linkage_mat)
                
        return c_assign
    
    
    def __cluster(self):
        """
        Clusters the data.
        
        :return:                cluster assignments, linkage matrix
        """
        c_ = np.arange(0, self.n)
        
        # initialize dists and linkage matrices
        dists = np.zeros((2 * self.n, 2 * self.n)); dists[:] = np.inf
        linkage = np.zeros((self.n - 1, 4))
        
        # merge clusters until we have 'n_cluster' clusters 
        for i in range(self.n - 1):
            if (self.n - i) == self.n_cluster:
                c_assign = np.copy(c_)
            # get all possible pairs of clusters
            c_ps = list(itertools.combinations(np.unique(c_), 2))
            
            # determine cluster pair to be merged
            # -----------------------------------------------------------------
            for c_p in c_ps:
                # calculate distance between cluster pair if necessary
                if dists[c_p[0], c_p[1]] == np.inf:
                    dists[c_p[0], c_p[1]] = self.__cluster_dist(
                        c_p[0], c_p[1], c_)
                  
            c_m = np.unravel_index(
                np.argmin(dists, axis=None), dists.shape)
                    
            # merge
            # -----------------------------------------------------------------
            c_new = np.max(c_) + 1
            c_[np.where(c_==c_m[0])] = c_new
            c_[np.where(c_==c_m[1])] = c_new
            
            # add entry to linkage matrix
            linkage[i, 0] = c_m[0]; linkage[i, 1] = c_m[1]
            linkage[i, 2] = dists[c_m[0], c_m[1]]
            linkage[i, 3] = len(np.where(c_==c_new)[0])
            
            # invalidate dists
            dists[c_m[0],:] = np.inf; dists[:,c_m[0]] = np.inf
            dists[c_m[1],:] = np.inf; dists[:,c_m[1]] = np.inf
        
        return c_assign, linkage
    
    
    def __dendrogram(self, linkage_mat):
        """
        Plots the dendrogram.
        
        :param linkage_mat:     linkage matrix
        """
        plt.figure()
        dendrogram(linkage_mat)
        plt.show()
    
    
    def __cluster_dist(self, c1, c2, c_assign):
        """
        Computes the cluster distance.
        
        :param c1:              data points in cluster 1
        :param c2:              data points in cluster 2
        :param c_assign:        current cluster assignments
        :return:                cluster distance
        """
        c1 = self.X[np.where(c_assign==c1)]
        c2 = self.X[np.where(c_assign==c2)]
                    
        # compute pairwise distances between all point in the two clusters
        dists = cdist(c1, c2, metric="euclidean")
        
        if self.method == "single_link":
            return np.min(dists)
        if self.method == "complete_link":
            return np.max(dists)
        