# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:04:58 2019

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import sys
import math
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Class MeanShift
# -----------------------------------------------------------------------------

class MeanShift():
    """
    Class MeanShift.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    
    
    def fit(self, X, bandwidth=1, min_dist=0.000001, dist_tolerance=0.1):
        """
        Fits the model to the training data.
        
        :param X:                   training data
        :param bandwidth:           kernel bandwidth
        :param min_dist:            minimum distance used to determine
                                    whether a data point still has to be shifted
        :param dist_tolerance:      distance tolerance used to determine the
                                    cluster assignments
        :return:                    cluster assignments
        """
        self.X = X
        self.bandwidth = bandwidth
        self.min_dist = min_dist
        self.dist_tolerance = dist_tolerance
        
        return self.__cluster()
        
        
    def __cluster(self):
        """
        Clusters the data points.
        
        :return:                    cluster assignments
        """
        shift_points = PointShifter().shift_points(self.X, self.bandwidth, self.min_dist)
        c_assign = ClusterCreator(dist_tolerance=self.dist_tolerance) \
            .cluster_points(shift_points)
        
        return c_assign
    

# -----------------------------------------------------------------------------
# Class PointShifter
# -----------------------------------------------------------------------------

class PointShifter():
    """
    Class MeanShift.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass


    def shift_points(self, points, kernel_bandwidth, min_dist):
        """
        Clusters the data points.
        
        :param points:              data points to cluster
        :param kernel_bandwidth:    band-width of the kernel
        :param min_dist:            minimum distance used to determine
                                    whether a data point still has to be shifted
        :return:                    clustered data points
        """
        # copy data points (these points will be shifted)
        shift_points = np.array(points)
        max_min_dist = 1
        # keep track of which points still have to be shifted
        still_shifting = [True] * points.shape[0]
        i = 0
        
        # while minimum distance is not reached
        while max_min_dist > min_dist:
            i += 1
            max_min_dist = 0
            
            for i in range(0, len(shift_points)):
                # check if data point still has to be shifted
                if not still_shifting[i]:
                    continue
                
                p_new = shift_points[i]
                p_new_start = p_new
                
                # shift point
                p_new = self.__shift_point(p_new, points, kernel_bandwidth)
                # calculate distance between old point and shifted version
                dist = euclidean_dist(p_new, p_new_start)
                
                if dist > max_min_dist:
                    max_min_dist = dist
                if dist < min_dist:
                    # minimum distance reached
                    still_shifting[i] = False
                shift_points[i] = p_new
                
                if i % 50 == 0:
                    self.__plot_data(shift_points, X_o=points)
            
        return shift_points.tolist()


    def __shift_point(self, point, points, bandwidth):
        """
        Shifts a data point.
        
        :param point:               data point to be shifted
        :param points:              data set
        :param bandwidth:           kernel bandwidth
        """
        points = np.array(points)

        # numerator
        point_weights = self.__gaussian_kernel(point - points, bandwidth)
        tiled_weights = np.tile(point_weights, [len(point), 1])
        
        # denominator
        denominator = sum(point_weights)
        shifted_point = np.multiply(tiled_weights.transpose(), points) \
            .sum(axis=0) / denominator
        
        return shifted_point
    
    
    def __gaussian_kernel(self, distance, bandwidth):
        """
        Implements the Gaussian kernel.
        
        :param distance:            distance between two points
        :param bandwidth:           bandwidth
        :return:                    kernel value
        """
        euclidean_distance = np.sqrt(((distance)**2).sum(axis=1))
        
        return (1 / (bandwidth * math.sqrt(2 * math.pi))) \
            * np.exp(-0.5 * ((euclidean_distance / bandwidth))**2)
    
        
    def __plot_data(self, X, X_o, title="Data"):
        """
        Plots the data.
        
        :param X:                   shifted data points
        :param X_o:                 original data points
        :param title:               title of the plot
        """
        plt.scatter(X_o[:,0], X_o[:,1], c="blue", alpha=0.7, edgecolors="k")
        plt.scatter(X[:,0], X[:,1], c="red", alpha=0.7, edgecolors="k")
        
        plt.title(title)
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        
        plt.show()
    
    
# -----------------------------------------------------------------------------
# Class ClusterCreator
# -----------------------------------------------------------------------------
    
class ClusterCreator():
    """
    Class ClusterCreator.
    """
    
    def __init__(self, dist_tolerance):
        """
        Constructor.
        
        :param dist_tolerance:      distance tolerance used to determine the
                                    cluster assignments
        """
        self.dist_tolerance = dist_tolerance
        
    
    def cluster_points(self, points):
        """
        Clusters the data points.
        
        :param points:              points to be clustered
        :return:                    cluster assignments
        """
        c_assign = []
        clusters = []
        group_index = 0
        
        # go over all points
        for point in points:
            nearest_cluster_index = self.__nearest_cluster(point, clusters)
            
            # create new cluster
            if nearest_cluster_index is None:
                clusters.append([point])
                c_assign.append(group_index)
                group_index += 1
                
            # add data point to existing cluster
            else:
                c_assign.append(nearest_cluster_index)
                clusters[nearest_cluster_index].append(point)
                
        return np.array(c_assign)


    def __nearest_cluster(self, point, clusters):
        """
        Determines the nearest cluster for a data point.
        
        :param point:               data point
        :param clusters:            list of clusters
        :return:
        """
        nearest_cluster_index = None
        index = 0
        
        for cluster in clusters:
            distance_to_cluster = self.__distance_to_cluster(point, cluster)
            if distance_to_cluster < self.dist_tolerance:
                nearest_cluster_index = index
            index += 1
            
        return nearest_cluster_index


    def __distance_to_cluster(self, point, cluster):
        """
        Calculates the distance to the group.
        
        :param point:               data point
        :param cluster:             cluster to which distance should be computed
        :return:                    distance to closest point in group
        """
        min_distance = sys.float_info.max
        
        for p in cluster:
            dist = euclidean_dist(point, p)
            if dist < min_distance:
                min_distance = dist
                
        return min_distance
    
    
# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def euclidean_dist(pointA, pointB):
    """
    Calculates the euclidean distance between two points.
    
    :param pointA:                  point a
    :param pointB:                  point b
    :return:                        euclidean distance
    """
    total = float(0)
    for dimension in range(0, len(pointA)):
        total += (pointA[dimension] - pointB[dimension])**2
    
    return math.sqrt(total)
