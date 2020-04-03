# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:59:14 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np


# -----------------------------------------------------------------------------
# class KMedoids
# -----------------------------------------------------------------------------

class KMedoids():
    """
    Class KMedoids.
    
    !!! EXPERIMENTAL !!!
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    
    
    def fit(self, X, k=3):
        """
        Fits a k-medoid model to the data.
        
        :param X:               training data
        :param k:               number of medoids / clusters
        :return:                cluster assignments
        """
        self.X = X
        self.n = self.X.shape[0]
        self.k = k
        
        c_assign = self.__cluster()
        
        return c_assign
        
        
    def __cluster(self):
        """
        Clusters the data.
        
        :return:                cluster assignments
        """
        # initialize the medoids
        medoids = self.X[
            np.random.choice(self.n, size=self.k, replace=False),:]
        
        converged = False
        
        while not converged:
            old_medoids = np.copy(medoids)    
            # compute distances to medoids
            dists = self.__compute_dist(medoids)
            c_assign = np.argmin(dists, axis=1)
            
            self.__update_medoids(medoids, c_assign)
            
            converged = self.__has_converged(old_medoids, medoids)
            
        return c_assign
            
    
    def __compute_dist(self, medoids):
        """
        Computes the distance of all data points to the medoids.
        
        :param medoids:         current medoids
        :return:                distances
        """
        dists = np.zeros((self.n, self.k))
        
        for i in range(self.n):
            dists[i,:] = np.linalg.norm(self.X[i,:] - medoids, axis=1)**2
            
        return dists
    
    
    def __update_medoids(self, medoids, c_assign):
        """
        Updates the medoids.
        
        :param c_assign:        cluster assignments
        """
        # go over all clusters
        for i in np.unique(c_assign):
            # compute current cost
            cost = np.sum(self.__compute_dist(medoids[i].reshape(1, -1)))
            
            # select another point from the cluster as medoid
            # and recompute the cost
            X_c = self.X[np.where(c_assign==i)]
            for x in X_c:
                cost_new = np.sum(self.__compute_dist(x.reshape(1, -1)))
                
                # if the cost is smaller than before -> swap medoid
                if cost_new < cost:
                    cost = cost_new
                    medoids[i] = x
                    
        return medoids
                
                
    def __has_converged(self, old_medoids, medoids):
        """
        Checks if the algorithm has converged.
        
        :param old_medoids:     old medoids
        :param medoids:         new medoids
        :return:                convergence flag
        """
        return set([tuple(x) for x in old_medoids]) \
            == set([tuple(x) for x in medoids])
            