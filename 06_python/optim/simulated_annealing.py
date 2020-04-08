# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:44:16 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


# -----------------------------------------------------------------------------
# Class SimulatedAnnealing
# -----------------------------------------------------------------------------

class SimulatedAnnealing():
    """
    Class SimulatedAnnealing.
    Only supports two-dimensional objective functions!
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    
    
    def optimize(self, f, n_iter, n_dim, s_lim=(-10, 10)):
        """
        Optimizes function f.
        
        :param f:               function to be optimized
        :param n_iter:          number of iterations
        :param n_dim:           number of dimensions
        :param s_lim:           search space limits
        :return:                optimum (minimum)
        """
        # vectorize function f
        f = np.vectorize(f, signature="(n)->()")
        # get initial state and cost
        S = [np.random.uniform(s_lim[0], s_lim[1], n_dim)]
        C = [f(S[-1])]
        
        # perform optimization
        for k in tqdm(range(n_iter)):
            frac = k / float(n_iter)
            # get current temperature
            T = self.__get_temp(frac)
            # get successor state
            s_new = self.__neighbour(S[-1], f, frac)
            c_new = f(s_new)
            
            # check if next state is accepted
            if self.__acceptance_prob(C[-1], c_new, T) > np.random.random():
                C.append(c_new)
                S.append(s_new)
                
        self.__plot(f, S, C, s_lim)
        
        return S[-1]
            
            
    def __get_temp(self, frac):
        """
        Computes the current temperature.
        
        :param frac:            fraction of current iteration to n_iter
        :return:                current temperature
        """
        return max(0.01, min(1, 1 - frac))
    
    
    def __neighbour(self, p, f, frac):
        """
        Gets a neighbor of x randomly.
        
        :param p:               point to get a neighbor for
        :param f:               function to be optimized
        :param frac:            fraction of current iteration to n_iter
        :return:                random neighbor of x
        """
        # create neighbors on a circle around point p
        r = 1 - frac
        A = np.linspace(0, 2 * np.pi, 60)[:-1]
        nbrs = np.asarray([[p[0] + r * np.cos(a), p[1] + r * np.sin(a)] for a in A])
        # select neighbor with minimal costs
        nbr = nbrs[np.argmin(f(nbrs))]
        
        return nbr
    
    
    def __acceptance_prob(self, c, c_new, temp):
        """
        Computes the acceptance probability of a new point.
        
        :param c:               old cost
        :param c_new:           new cost
        :param temp:            current temperature
        :return:                acceptance probability
        """
        if c_new < c:
            return 1
        else:
            return np.exp(-(c_new - c) / temp)
    
    
    def __plot(self, f, S, C, s_lim):
        """
        Plots the results.
        
        :param f:               function to be optimized
        :param S:               array of states
        :param C:               array of costs
        """
        # plot costs
        # ---------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10.0, 10.0))
        plt.plot(C)
        
        plt.show()
        
        # plot function
        # ---------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10.0, 10.0))
        x1, x2 = np.meshgrid(
            np.linspace(s_lim[0], s_lim[1], 300),
            np.linspace(s_lim[0], s_lim[1], 300)
        )
        
        ax.set_xlim((s_lim[0], s_lim[1]))
        ax.set_ylim((s_lim[0], s_lim[1]))

        # create contour plot
        cf = ax.contourf(
            x1, x2, f(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape[0],-1),
            levels=30, zorder=5)
        ax.contour(cf, colors="k", zorder=5)
        
        # plot states visited
        S = np.concatenate(S).reshape(-1, 2)
        ax.scatter(S[:,0], S[:,1], c="r", s=75, zorder=10)
        
        # connect states
        for i in range(S.shape[0] - 1):
            plt.plot([S[i, 0], S[i + 1, 0]], [S[i, 1], S[i + 1, 1]], "r", zorder=9)
            
        plt.show()
            