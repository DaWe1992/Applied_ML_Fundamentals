# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:17:54 2020
Cf. https://www.ritchievink.com/blog/2018/05/18/algorithm-breakdown-affinity-propagation/

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle


# -----------------------------------------------------------------------------
# Class AffinityPropagation
# -----------------------------------------------------------------------------

class AffinityPropagation():
    """
    Class AffinityPropagation.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    
    
    def fit(self, X, damping=0.9, n_iter=10, plot=True):
        """
        Fits an affinity propagation model to the data.
        
        :param X:               training data
        :param damping:         damping factor
        :param n_iter:          number of iterations
        :param plot:            flag whether to plot the results
        :return:                cluster assignments
        """
        self.X = X
        self.n = X.shape[0]
        self.damping = damping
        
        # initialize matrices
        self.A, self.R, self.S = self.__create_matrices()
        
        np.fill_diagonal(self.S, -1000)
        
        # perform training iterations
        for i in range(n_iter):
            # update R
            self.__update_r()
            # update A
            self.__update_a()
            # get cluster assignments
            c_assign = np.argmax(self.A + self.R, axis=1)
    
            # plot intermediate results
            if i % 5 == 0:
                self.__plot_iteration(c_assign)
                
        return c_assign
    
    
    def __create_matrices(self):
        """
        Initializes the required matrices A, R, and S.
        
        :return:                availability matrix A, responsibility matrix R
                                and similarity matrix S
        """
        S = np.zeros((self.n, self.n))
        R = np.array(S)
        A = np.array(S)
        
        # compute similarity matrix
        for i in range(self.n):
            for j in range(self.n):
                S[i, j] = -((self.X[i] - self.X[j])**2).sum()
                
        return A, R, S
    
    
    def __update_r(self):
        """
        Updates the responsibility matrix R.
        """
        V = self.S + self.A
        rows = np.arange(self.n)
        
        # a point does not send messages to itself
        np.fill_diagonal(V, -np.inf)

        # max values
        # ---------------------------------------------------------------------
        idx_max = np.argmax(V, axis=1)
        first_max = V[rows, idx_max]

        V[rows, idx_max] = -np.inf
        second_max = V[rows, np.argmax(V, axis=1)]

        # broadcast the maximum value per row over all columns
        max_matrix = np.zeros_like(self.R) + first_max[:, None]
        max_matrix[rows, idx_max] = second_max

        # update matrix R
        self.R = self.R * self.damping + (1 - self.damping) * (self.S - max_matrix)


    def __update_a(self):
        """
        Updates the availability matrix A.
        """
        k_k_idx = np.arange(self.n)
        # set a(i, k)
        V = np.array(self.R)
        V[V < 0] = 0
        np.fill_diagonal(V, 0)
        V = V.sum(axis=0)
        V += self.R[k_k_idx, k_k_idx]

        V = np.ones(self.A.shape) * V

        # for every column k, subtract the positive value of k
        V -= np.clip(self.R, 0, np.inf)
        V[V > 0] = 0
        
        # set diagonal entries
        V_ = np.array(self.R)
        np.fill_diagonal(V_, 0)

        V_[V_ < 0] = 0

        V[k_k_idx, k_k_idx] = V_.sum(axis=0)
        
        # update matrix A
        self.A = self.A * self.damping + (1 - self.damping) * V
        
        
    def __plot_iteration(self, c_assign):
        """
        Plots the result of one iteration.
        
        :param c_assign:        cluster assignments
        """
        fig, ax = plt.subplots(figsize=(12.00, 7.00))
        
        # draw gridlines
        ax.grid(b=True, which="major", color="gray", \
            linestyle="--", zorder=5)
    
        exemplars = np.unique(c_assign)
        colors = dict(zip(exemplars, cycle("bgrcmyk")))
        
        # go over all data points
        for i in range(len(c_assign)):
            x1 = self.X[i][0]
            x2 = self.X[i][1]
            
            # data point is an exemplar
            if i in exemplars:
                exemplar = i
                edge = "k"; ms = 10; z = 10
            else:
                # data point is not an exemplar
                exemplar = c_assign[i]
                ms = 3; edge = None; z = 9
                # plot line from point to exemplar
                ax.plot([x1, self.X[exemplar][0]],
                        [x2, self.X[exemplar][1]], c=colors[exemplar])
            
            # plot data point
            ax.plot(x1, x2, "o", markersize=ms, 
                markeredgecolor=edge, c=colors[exemplar], zorder=z)
        
        plt.show()
        