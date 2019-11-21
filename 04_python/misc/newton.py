# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 08:08:45 2019

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def optimize():
    """
    Optimizes function x^2 + y^2
    """
    # function x^2 + y^2
    # (optimum at [0, 0])
    
    # initial guess
    x = np.asarray([5, 28])
    
    # compute Hessian and gradient
    H = np.asarray([[2, 0], [0, 2]])
    g = np.asarray([2*x[0], 2*x[1]])
    
    # compute update
    delta_x = -np.linalg.inv(H) @ g
    
    # update x
    x = x + delta_x
    print("Optimum: {0}".format(x))
    
    
def plot():
    """
    Plots function.
    """
    def f(x, y):
        return x**2 + y**2
    
    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)
    
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    plt.figure()
    ax = plt.axes(projection="3d")
    ax.contour3D(X, Y, Z, 50, cmap="binary")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z");
    
    plt.show()
    

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    optimize()
    plot()
    