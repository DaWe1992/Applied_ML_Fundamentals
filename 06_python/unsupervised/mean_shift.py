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
# Configuration
# -----------------------------------------------------------------------------

MIN_DISTANCE = 0.000001                 # for cluster algorithm
GROUP_DISTANCE_TOLERANCE = 0.1          # for point grouper

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def euclidean_dist(pointA, pointB):
    """
    Calculates the euclidean distance between two points.
    
    :param pointA:              point a
    :param pointB:              point b
    :return:                    euclidean distance
    """
    # check dimensionality
    if(len(pointA) != len(pointB)):
        raise Exception("expected point dimensionality to match")
    
    total = float(0)
    for dimension in range(0, len(pointA)):
        total += (pointA[dimension] - pointB[dimension])**2
    
    return math.sqrt(total)


def gaussian_kernel(distance, bandwidth):
    """
    Implements the Gaussian kernel.
    
    :param distance:            distance between two points
    :param bandwidth:           bandwidth
    :return:                    kernel value
    """
    euclidean_distance = np.sqrt(((distance)**2).sum(axis=1))
    return (1 / (bandwidth * math.sqrt(2 * math.pi))) \
        * np.exp(-0.5 * ((euclidean_distance / bandwidth))**2)


def multivariate_gaussian_kernel(distances, bandwidths):
    """
    Implements the multivariate Gaussian kernel.
    
    :param distance:            distance between two points
    :param bandwidth:           bandwidth
    :return:                    kernel value
    """
    # Number of dimensions of the multivariate gaussian
    dim = len(bandwidths)
    # covariance matrix
    cov = np.multiply(np.power(bandwidths, 2), np.eye(dim))
    # compute multivariate gaussian (vectorized implementation)
    exponent = -0.5 * np.sum(np.multiply(np.dot(
        distances, np.linalg.inv(cov)), distances), axis=1)
    return (1 / np.power((2 * math.pi), (dim / 2)) \
        * np.power(np.linalg.det(cov), 0.5)) * np.exp(exponent)


# -----------------------------------------------------------------------------
# Class PointGrouper
# -----------------------------------------------------------------------------
    
class PointGrouper(object):
    """
    Class PointGrouper.
    """
    
    def group_points(self, points):
        """
        Groups the points.
        
        :param points:          grouped points
        """
        group_assignment = []
        groups = []
        group_index = 0
        
        for point in points:
            nearest_group_index = self.__determine_nearest_group(point, groups)
            if nearest_group_index is None:
                # create new group
                groups.append([point])
                group_assignment.append(group_index)
                group_index += 1
            else:
                group_assignment.append(nearest_group_index)
                groups[nearest_group_index].append(point)
                
        return np.array(group_assignment)


    def __determine_nearest_group(self, point, groups):
        """
        Determines the nearest group.
        
        :param point:           data point
        :param groups:          list of groups
        :return:
        """
        nearest_group_index = None
        index = 0
        
        for group in groups:
            distance_to_group = self.__distance_to_group(point, group)
            if distance_to_group < GROUP_DISTANCE_TOLERANCE:
                nearest_group_index = index
            index += 1
            
        return nearest_group_index


    def __distance_to_group(self, point, group):
        """
        Calculates the distance to the group.
        
        :param point:           data point
        :param group:           group to which distance should be computed
        :return:                distance to closest point in group
        """
        min_distance = sys.float_info.max
        
        for pt in group:
            dist = euclidean_dist(point, pt)
            if dist < min_distance:
                min_distance = dist
                
        return min_distance


# -----------------------------------------------------------------------------
# Class MeanShift
# -----------------------------------------------------------------------------

class MeanShift(object):
    """
    Class MeanShift.
    """
    
    def __init__(self, kernel=gaussian_kernel):
        """
        Constructor.
        
        :param kernel:              kernel fuction to use for kde
        """
        if kernel == "multivariate_gaussian":
            kernel = multivariate_gaussian_kernel
        self.kernel = kernel


    def cluster(self, points, kernel_bandwidth):
        """
        Clusters the data points.
        
        :param points:              data points to cluster
        :param kernel_bandwidth:    band-width of the kernel
        :return:                    clustered data points
        """
        # copy data points (these points will be shifted)
        shift_points = np.array(points)
        max_min_dist = 1
        # keep track of which points still have to be shifted
        still_shifting = [True] * points.shape[0]
        i = 0
        
        # while minimum distance is not reached
        while max_min_dist > MIN_DISTANCE:
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
                if dist < MIN_DISTANCE:
                    # minimum distance reached
                    still_shifting[i] = False
                shift_points[i] = p_new
                if i % 50 == 0:
                    self.plot_data(shift_points, X_o=points)
            
        point_grouper = PointGrouper()
        group_assignments = point_grouper.group_points(shift_points.tolist())
        
        self.plot_data_clusters(points, group_assignments)
        
        return MeanShiftResult(points, shift_points, group_assignments)


    def __shift_point(self, point, points, kernel_bandwidth):
        """
        Shifts data point.
        from http://en.wikipedia.org/wiki/Mean-shift
        """
        points = np.array(points)

        # numerator
        point_weights = self.kernel(point - points, kernel_bandwidth)
        tiled_weights = np.tile(point_weights, [len(point), 1])
        
        # denominator
        denominator = sum(point_weights)
        shifted_point = np.multiply(tiled_weights.transpose(), points).sum(axis=0) / denominator
        
        return shifted_point
    
        
    def plot_data(self, X, X_o, title="Data"):
        """
        Plots the data.
        
        :param X:               data to be plotted
        :param title:           title of the plot
        """
        plt.scatter(X_o[:,0], X_o[:,1], c="blue", alpha=0.7, edgecolors="black")
        plt.scatter(X[:,0], X[:,1], c="red", alpha=0.7, edgecolors="black")
        
        plt.title(title)
        plt.xlabel("Feature 0")
        plt.ylabel("Feature 1")
        
        plt.show()
    
    
    def plot_data_clusters(self, X, clusters, title="Data"):
        """
        Plots the data.
        
        :param X:               data to be plotted
        :param title:           title of the plot
        """
        plt.scatter(X[:,0], X[:,1], c=clusters, alpha=0.8, edgecolors="black")
        
        plt.title(title)
        plt.xlabel("Feature 0")
        plt.ylabel("Feature 1")
        
        plt.show()


# -----------------------------------------------------------------------------
# Class MeanShiftResult
# -----------------------------------------------------------------------------
        
class MeanShiftResult:
    """
    Class MeanShiftResult.
    Wrapper class for results.
    """
    
    def __init__(self, original_points, shifted_points, cluster_ids):
        """
        Constructor.
        
        :param original points:     original data points
        :param shifted points:      shifted data points
        :param cluster_ids:         cluster ids
        """
        self.original_points = original_points
        self.shifted_points = shifted_points
        self.cluster_ids = cluster_ids
        
        
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
        
def main():
    """
    Main function.
    """
    reference_points = np.genfromtxt("../data/data.csv", delimiter=",")
    mean_shifter = MeanShift()
    mean_shift_result = mean_shifter.cluster(reference_points, kernel_bandwidth=3)

    print("Original Point     Shifted Point  Cluster ID")
    print("--------------------------------------------")
    for i in range(len(mean_shift_result.shifted_points)):
        original_point = mean_shift_result.original_points[i]
        converged_point = mean_shift_result.shifted_points[i]
        cluster_assignment = mean_shift_result.cluster_ids[i]
        print("(%5.2f,%5.2f)  ->  (%5.2f,%5.2f)  cluster %i" \
              % (original_point[0], original_point[1], converged_point[0], converged_point[1], cluster_assignment))
        

# if __name__ == "__main__":
#     main()