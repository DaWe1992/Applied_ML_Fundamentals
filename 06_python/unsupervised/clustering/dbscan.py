# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 11:08:03 2018
@author: Daniel Wehner

Implementation of DBSCAN clustering
(Density-Based Spatial Clustering of Applications with Noise)

and OPTICS
(Ordering Points to Identify the Clustering Structure)
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import math
import numpy as np
import matplotlib.pyplot as plt

from scipy import spatial
from heapq import heappush, heappop


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
        Fits the model to the training data.
        
        :param X:               training data
        :return:                cluster assignments
        """
        self.X = []
        
        # create data point objects
        for x in X:
            self.X.append(DataPoint(x))
            
        return self.__cluster()
                    
    
    def __cluster(self):
        """
        Cluster the data.
        
        :return:                cluster assignments
        """
        # cluster counter
        c = -1
        
        for x in self.X:
            
            # check if data point is already part of cluster
            if x.has_label(): continue
                
            # find neighbors of data point
            N = get_neighbors(self.X, x, self.eps)
                
            # NOISE
            # -----------------------------------------------------------------
            # classify data point as noise (-1),
            # if it has not enough neighbors
            if len(N) < self.min_pts:
                x.set_label(-1)
                continue
                
            # NOT NOISE
            # -----------------------------------------------------------------
            # increment cluster counter
            c += 1
            x.set_label(c)
                
            # initialize seed set for cluster
            S = [n for n in N if n not in [x]]
            
            i = 0
            while i != len(S):
                s = S[i]
                i += 1
                
                if s.get_label() == -1: s.set_label(c)
                if s.has_label(): continue
                s.set_label(c)
                
                # get neighbors for new point and add them to S
                N = get_neighbors(self.X, s, self.eps)
                if len(N) >= self.min_pts:
                    S = S + [n for n in N if n not in S]
                    
        return self.__get_cluster_list()
        
    
    def __get_cluster_list(self):
        """
        Extracts the clusters found for the data.
        
        :return:                list of clusters
        """
        clusters = []
        
        for x in self.X:
            clusters.append(x.get_label())
            
        return clusters
        

# -----------------------------------------------------------------------------
# Class OPTICS
# -----------------------------------------------------------------------------

class OPTICS():
    """
    Class OPTICS.
    Cf. https://www.dbs.ifi.lmu.de/Publikationen/Papers/OPTICS.pdf
    """
    
    def __init__(self, eps, eps_, min_pts, plot_reach=True):
        """
        Constructor.
        
        :param eps:             minimum distance to neighbor
        :param eps_:            reachability distance threshold for cluster
                                extraction
        :param min_pts:         minimum number of data points in the neighborhood
                                in order for a data point to be considered
                                a core point/density-reachable point
        :param plot_reach:      flag indicating whether to plot the
                                reachability values
        """
        if eps_ > eps:
            raise ValueError("Precondition: eps_ <= eps!")
            
        self.eps = eps
        self.eps_ = eps_
        self.min_pts = min_pts
        self.plot_reach = plot_reach
    
    
    def fit(self, X):
        """
        Fits the OPTICS model to the data.
        
        :param X:               training data
        :return:                cluster assignments
        """
        self.X = []
        
        # create data point objects
        for x in X:
            self.X.append(DataPoint(x))
        
        # start OPTICS algorithm
        ordered_list = []
        for x in self.X:
            if not x.processed:
                self.__expand_cluster_order(x, ordered_list)
        
        print([x.r_dist for x in ordered_list])
        ordered_list[0] = ordered_list[1]
        self.__extract_clusters(ordered_list)
        
        # plot reachability values if specified
        if self.plot_reach:
            self.__plot_reachability(ordered_list)
        
        return [x.get_label() for x in self.X]
        
        
    def __expand_cluster_order(self, x, ordered_list):
        """
        Clusters the data.
        
        :param x:               data point object
        :param ordered_list:    ordered list (contains result)
        """
        nbrs = get_neighbors(self.X, x, self.eps)
        x.processed = True
        x.r_dist = None
        x.c_dist = self.__c_dist(x)
        ordered_list.append(x)
        
        if x.c_dist is not None:
            seeds = PriorityQueue()
            self.__update(nbrs, x, seeds)
            
            while True:
                if len(seeds.list) == 0:
                    break
                x_ = seeds.pop()
                nbrs_ = get_neighbors(self.X, x_, self.eps)
                x_.processed = True
                x_.c_dist = self.__c_dist(x_)
                ordered_list.append(x_)
                
                if x_.c_dist is not None:
                    self.__update(nbrs_, x_, seeds)
                            
                           
    def __update(self, nbrs, x, seeds):
        """
        Updates the reachability distances for all neighbors.
        
        :param nbrs:            list of neighbors
        :param x:               data point object
        :param seeds:           seed order priority queue
        """
        # go over all neighbors
        for nbr in nbrs:
            if not nbr.processed:
                r_dist_new = self.__r_dist(x, nbr)
                
                # element is already in seeds
                if nbr.r_dist is None:
                    nbr.r_dist = r_dist_new
                    seeds.insert(r_dist_new, nbr)
                # element is not in seeds
                else:
                    if r_dist_new < nbr.r_dist:
                        nbr.r_dist = r_dist_new
                        seeds.update(r_dist_new, nbr)
                        
                        
    def __extract_clusters(self, ordered_list):
        """
        Extracts the clusters.
        
        :param ordered_list:    ordered list (contains the result)
        """
        cluster_id = 0
        
        # go over all data points
        for x in ordered_list:
            if x.r_dist > self.eps_:
                # increase cluster_id
                if x.c_dist <= self.eps_:
                    cluster_id += 1
                    x.set_label(cluster_id)
                else:
                    x.set_label(-1)
                    
            else:
                x.set_label(cluster_id)
        
                
    def __c_dist(self, x):
        """
        Calculates the core distance of a point.
        
        :return:                core distance
        """
        # compute min_pts-th nearest neighbor
        ds, _ = spatial.KDTree(np.asarray([p.x for p in self.X])) \
            .query(x.x, k=self.min_pts)
        # compute core distance
        core_dist = ds[-1] if ds[-1] < self.eps else None
        
        return core_dist
    
    
    def __r_dist(self, x, x_):
        """
        Computes the reachability distance.
        
        :param x:
        :param x_:
        :return:                reachability distance
        """
        return max(x.c_dist, dist(x, x_))
    
    
    def __plot_reachability(self, ordered_list):
        """
        Plots the reachability values.
        
        """
        fig, ax = plt.subplots(figsize=(12.00, 7.00))
        ax.set_title("Reachability Plot", fontsize=18, fontweight="demi")
        
        # axis labels
        ax.set_xlabel("data point", fontsize=12)
        ax.set_ylabel("reachability score", fontsize=12)
        
        # draw major grid
        ax.grid(b=True, which="major", color="gray", \
            linestyle="--", zorder=5)
            
        x_range = np.arange(0, len(self.X))
        ax.scatter(
            x_range,
            np.asarray([e.r_dist for e in ordered_list]),
            c=[x.get_label() for x in ordered_list],
            zorder=10, edgecolors="k", s=75
        )
        ax.plot(x_range, [self.eps_] * len(x_range), "r--")
        
        plt.show()
    
  
# -----------------------------------------------------------------------------
# PriorityQueue
# -----------------------------------------------------------------------------
        
class PriorityQueue:
    """
    Class PriorityQueue.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        self.list = []
    
    
    def insert(self, key, value):
        """
        Inserts an element into the list.
        
        :param key:             key used for sorted insertion
        :param value:           value to be inserted into the list
        """
        heappush(self.list, (key, value))
    
    
    def update(self, key, value):
        """
        Updates a list element.
        
        :param key:             key used for sorted insertion
        :param value:           value to be inserted into the list
        """
        # get index of element to be updated
        idx = sum([(elem[1] == value) * i for (i, elem) in enumerate(self.list)])
        # delete element at position idx
        del self.list[idx]
        # newly insert element
        self.insert(key, value)
        
        
    def pop(self):
        """
        Gets the first element from the queue.
        
        :return:                first element of queue
        """
        return heappop(self.list)[1]
    
 
# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
    
def get_neighbors(X, x_, eps):
    """
    Finds all points in the neighborhood.
    
    :param X:               training data
    :param x_:              data point
    :param eps:             epsilon parameter
    :return:                list of all neighbors
    """
    nbrs = []
    
    for x in X:
        if(dist(x, x_) <= eps):
            nbrs.append(x)
            
    return nbrs


def dist(x1, x2):
    """
    Calculates the distance between points x and y.
    (Using euclidean distance)
    
    :param x:               point x1
    :param y:               point x2
    :return:                distance between points x1 and x2
    """
    return math.sqrt(
        math.pow(x1.get_x()[0] - x2.get_x()[0], 2) +
        math.pow(x1.get_x()[1] - x2.get_x()[1], 2))
    
    
# -----------------------------------------------------------------------------
# Class DataPoint
# -----------------------------------------------------------------------------
        
class DataPoint:
    """
    Class DataPoint.
    """
    
    def __init__(self, x):
        self.x = x
        self.y = None
        self.r_dist = None      # only used in OPTICS
        self.c_dist = None      # only used in OPTICS
        self.processed = False  # only used in OPTICS
        
        
    def __lt__(self, other):
        """
        'Less than' function used for comparison
        with other objects of this type.
        """
        return hash(self) < hash(other)
        
        
    def get_x(self):
        return self.x
    
    
    def set_label(self, y):
        self.y = y
        
        
    def get_label(self):
        return self.y
    
    
    def has_label(self):
        return self.y != None
    