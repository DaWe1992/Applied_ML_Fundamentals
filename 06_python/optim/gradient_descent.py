# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:13:03 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np


# -----------------------------------------------------------------------------
# Class GradientDescent
# -----------------------------------------------------------------------------

class GradientDescent():
    """
    Class GradientDescent.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
        
        
    def optimize(self, f, x_start, alpha=0.1, n_iter=1000, delta=0.01):
        """
        Optimizes function f.
        
        :param f:                   function to be optimized
        :param x_start:             initial guess
        :param alpha:               learning rate
        :param n_iter:              number of iterations
        :param delta:               delta for finite differences
        :return:                    optimum
        """
        x = np.asarray(x_start)
        dim = len(x)
        
        # gradient vector
        grad = np.zeros(dim)
        
        # perform training iterations
        for i in range(n_iter):
            # compute the gradient per dimension
            for j in range(dim):
                v_delta = np.zeros(dim)
                v_delta[j] = delta
                
                # compute the approximate gradient (finite differences)
                grad[j] = f(x + v_delta) - f(x - v_delta)
                
            # gradient descent update
            x -= alpha * grad
            
        return x
    
  
# -----------------------------------------------------------------------------
# Functions to be optimized
# -----------------------------------------------------------------------------
        
def f(x):
    return (x - 2)**2


def g(x):
    return (x[0] - 1)**2 + (x[1] + 5)**2
    
    
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
    
if __name__ == "__main__":
    gd = GradientDescent()
    x_opt = gd.optimize(g, [0.0, 0.0], n_iter=10000)
    print(x_opt)