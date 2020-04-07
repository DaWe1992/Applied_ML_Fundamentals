# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:25:49 2020
Nelder Mead Simplex algorithm.
Cf. https://codesachin.wordpress.com/2016/01/16/nelder-mead-optimization/

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


# -----------------------------------------------------------------------------
# Class NelderMead
# -----------------------------------------------------------------------------

class NelderMead():
    """
    Class NelderMead.
    Only supports two-dimensional points
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass

    
    def optimize(self, f, n_iter=100, alpha=1.0, beta=0.5, gamma=2.0, delta=0.5):
        """
        Optimizes function f.
        
        :param f:               function to be optimized
        :param n_iter:          number of iterations
        :param alpha:           reflection parameter
        :param beta:            contraction parameter
        :param gamma:           expansion parameter
        :param delta:           shrinkage parameter
        :return:                optimum (minimum)
        """
        # initialize simplex
        X = self.__init_simplex()
        # vectorize function f
        f = np.vectorize(f, signature="(n)->()")
        # plot initial simplex
        self.__plot(f, X)
        
        # for specified number of iterations do:
        for k in tqdm(range(n_iter)):
            # sort simplex points from lowest to highest:
            f_X = -f(X)
            X = X[np.argsort(f_X)]
            # compute centroid c (ignore worst point)
            c = np.mean(X[1:], axis=0)
            
            # try reflection: x_r = c + alpha * (c - x_0)
            # -----------------------------------------------------------------
            x_r = c + alpha * (c - X[0])
            f_x_r = -f(x_r)
            if f_X[1] < f_x_r and f_x_r <= f_X[2]:
                # replace x_0 with x_r
                X[0] = x_r
            
            # try expansion:
            # -----------------------------------------------------------------
            elif f_x_r > f_X[2]:
                x_e = c + gamma * (x_r - c)
                
                # replace x_0 with the better of two points x_e and x_r
                X[0] = x_e if -f(x_e) < f_x_r else x_r
            
            # try contraction:
            # -----------------------------------------------------------------
            elif f_x_r < f_X[1]:
                x_c = c + beta * (X[0] - c)
            
                if -f(x_c) > f_X[0]:
                    # replace x_0 with x_c
                    X[0] = x_c
            
                # try shrink contraction:
                # -------------------------------------------------------------
                else:
                    # redefine the entire simplex
                    # only keep the best point x_2
                    X[0] = X[2] + delta * (X[0] - X[2])
                    X[1] = X[2] + delta * (X[1] - X[2])
                    
            self.__plot(f, X)
                    
        return X[2]
            
            
    def __init_simplex(self):
        """
        Initializes the simplex.
        
        :return:                initialized simplex
        """
        return np.asarray([
            [-1.50, -1.50],
            [-1.00, -1.25],
            [-1.25, -1.75]
        ])
    
    
    def __plot(self, f, X):
        """
        Plots the optimization progress.
        
        :param f:               function to be optimized
        :param X:               simplex points
        """
        x1, x2 = np.meshgrid(
            np.linspace(-2, 2, 300),
            np.linspace(-2, 2, 300)
        )
        
        fig, ax = plt.subplots(figsize=(10.0, 10.0))
        ax.set_xlim((-2, 2))
        ax.set_ylim((-2, 2))

        # create contour plot
        cf = ax.contourf(
            x1, x2, f(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape[0],-1),
            levels=30, zorder=5)
        ax.contour(cf, colors="k", zorder=5)
        
        # plot simplex points
        ax.scatter(X[:,0], X[:,1], c="r", zorder=10)
        # plot simplex point connections
        plt.plot([X[0][0], X[1][0]], [X[0][1], X[1][1]], "r", zorder=9)
        plt.plot([X[0][0], X[2][0]], [X[0][1], X[2][1]], "r", zorder=9)
        plt.plot([X[1][0], X[2][0]], [X[1][1], X[2][1]], "r", zorder=9)
        
        plt.show()
        