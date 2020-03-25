# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 11:08:03 2018
@author: Daniel Wehner

Implementation of DBSCAN clustering
(Density-Based Spatial Clustering of Applications with Noise)
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import math
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Class DBSCAN
# -----------------------------------------------------------------------------

class DBSCAN():
    """
    Class DBSCAN.
    (Density-Based Spatial Clustering of Applications with Noise)
    """
    
    def __init__(self, eps, min_pts):
        """
        Constructor.
        
        :param eps:             minimum distance to neighbor
        :param min_pts:         minimum number of data points in the neighborhood
                                    in order for a data point to be considered
                                    a core point/density-reachable point
        """
        self.eps = eps
        self.min_pts = min_pts
        
        
    def fit(self, X):
        """
        Fit to training data.
        
        :param X:               training data
        """
        self.X = []
        
        # create data point objects
        for x in X:
            self.X.append(DataPoint(x))
            
        return self.cluster()
                    
    
    def cluster(self):
        """
        Cluster the data.
        
        :return:                list of clusters
        """
        # cluster counter
        c = 0
        
        for x in self.X:
            
            # check if data point is already part of cluster
            if x.hasLabel(): continue
                
            # find neighbors of data point
            N = self.__rangeQuery(x)
                
            # NOISE
            # -----------------------------------------------------------------
            # classify data point as noise (-1),
            # if it has not enough neighbors
            if len(N) < self.min_pts:
                x.setLabel(-1)
                continue
                
            # NOT NOISE
            # -----------------------------------------------------------------
            # increment cluster counter
            c += 1
            x.setLabel(c)
                
            # initialize seed set for cluster
            S = [n for n in N if n not in [x]]
            
            i = 0
            while i != len(S):
                s = S[i]
                i += 1
                
                if s.getLabel() == -1: s.setLabel(c)
                if s.hasLabel(): continue
                s.setLabel(c)
                
                # get neighbors for new point and add them to S
                N = self.__rangeQuery(s)
                if len(N) >= self.min_pts:
                    S = S + [n for n in N if n not in S]
                    
        return self.__get_cluster_list()
    
    
    def __rangeQuery(self, y):
        """
        Finds all points in the neighborhood.
        
        :param y:               data point
        :return:                list of all neighbors
        """
        nbrs = []
        
        for x in self.X:
            if(self.__dist(x, y) <= self.eps):
                nbrs.append(x)
                
        return nbrs

    
    def __dist(self, x1, x2):
        """
        Calculates the distance between points x and y.
        (Using euclidean distance)
        
        :param x:               point x1
        :param y:               point x2
        :return:                distance between points x1 and x2
        """
        return math.sqrt(
            math.pow(x1.getX()[0] - x2.getX()[0], 2) +
            math.pow(x1.getX()[1] - x2.getX()[1], 2))
        
    
    def __get_cluster_list(self):
        """
        Extracts the clusters found for the data.
        
        :return:                list of clusters
        """
        clusters = []
        
        for x in self.X:
            clusters.append(x.getLabel())
            
        return clusters
        

# -----------------------------------------------------------------------------
# Class DataPoint
# -----------------------------------------------------------------------------
        
class DataPoint:
    """
    Class DataPoint.
    """
    
    def __init__(self, X):
        self.X = X
        self.y = None
        
        
    def getX(self):
        return self.X
    
    
    def setLabel(self, y):
        self.y = y
        
        
    def getLabel(self):
        return self.y
    
    
    def hasLabel(self):
        return self.y != None
    

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
        
def plot_clusters(X, clusters, title):
    """
    Plots the clusters.
    
    :param X:           data points to be visualized
    :param clusters:    clusters
    :param title:       title of the plot
    """
    plt.scatter(X[:,0], X[:,1], c=clusters, s=60)
    plt.title(title)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()
    

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
    
if __name__ == "__main__":
    """
    Main entry point.
    """
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    
    # scale data
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    
    dbscan = DBSCAN(eps=0.5, min_pts=3)
    c_assign = dbscan.fit(X_scaled)
    print(c_assign)
    
    plot_clusters(X_scaled, c_assign, "DBSCAN")
    