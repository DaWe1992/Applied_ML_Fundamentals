# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:44:16 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np

from tqdm import tqdm


# -----------------------------------------------------------------------------
# Class SimulatedAnnealing
# -----------------------------------------------------------------------------

class SimulatedAnnealing():
    """
    Class SimulatedAnnealing.
    
    !!! EXPERIMENTAL !!!
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
        self.j = 0
        # vectorize function f
        f = np.vectorize(f, signature="(n)->()")
        # get initial state and cost
        s = np.random.uniform(s_lim[0], s_lim[1], n_dim)
        c = f(s)
        
        # perform optimization
        for k in tqdm(range(n_iter)):
            frac = k / float(n_iter)
            # get current temperature
            T = self.__get_temp(frac)
            # get successor state
            s_new = self.__neighbour(s, f, frac)
            c_new = f(s_new)
            
            # check if next state is accepted
            if self.__acceptance_prob(c, c_new, T) > np.random.random():
                s, c = s_new, c_new
                
        print(self.j)
        return s
            
            
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
        A = np.linspace(0, 2 * np.pi, 60)
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
            self.j += 1
            return np.exp(-(c_new - c) / temp)
    