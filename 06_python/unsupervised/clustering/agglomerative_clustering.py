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

from scipy.spatial.distance import cdist


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
    
    
    def fit(self, X, n_cluster, method="complete_link"):
        """
        Fits an agglomerative clustering model to the data.
        
        :param X:               training data
        :param n_cluster:       number of clusters
        :param method:          clustering method, e.g.:
                                    - single_link
                                    - complete_link
        :return:                cluster assignments
        """
        self.X = X
        self.n_cluster = n_cluster
        self.method = method
        
        c_assign = self.__cluster()
        
        return c_assign
    
    
    def __cluster(self):
        """
        Clusters the data.
        
        :return:                cluster assignments
        """
        c_assign = np.arange(0, self.X.shape[0])
        n_curr_cluster = self.X.shape[0]
        
        # merge clusters until we have 'n_cluster' clusters 
        while n_curr_cluster > self.n_cluster:
            # get all possible pairs of clusters
            c_pairs = list(itertools.combinations(np.unique(c_assign), 2))
            
            # determine cluster pair to be merged
            # -----------------------------------------------------------------
            best_dist = np.inf
            best_pair = None
            for c_pair in c_pairs:
                # calculate distance between cluster pair
                dist = self.__cluster_dist(
                    self.X[np.where(c_assign==c_pair[0])],
                    self.X[np.where(c_assign==c_pair[1])])
                # update best cluster pair to be merged
                if dist < best_dist:
                    best_dist = dist
                    best_pair = c_pair
                    
            # merge
            # -----------------------------------------------------------------
            c_assign[np.where(c_assign==best_pair[1])] = best_pair[0]
                    
            n_curr_cluster -= 1
            
        return c_assign
    
    
    def __cluster_dist(self, c1, c2):
        """
        Computes the cluster distance.
        
        :param c1:              data points in cluster 1
        :param c2:              data points in cluster 2
        :return:                cluster distance
        """
        # compute pairwise distances between all point in the two clusters
        dists = cdist(c1, c2, metric="euclidean")
        
        if self.method == "single_link":
            return np.min(dists)
        if self.method == "complete_link":
            return np.max(dists)
        