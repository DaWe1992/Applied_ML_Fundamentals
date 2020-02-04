# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:26:23 2020

@author: Daniel Wehner
@see: https://xavierbourretsicotte.github.io/loess.html
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg


# -----------------------------------------------------------------------------
# Class LOWESS
# -----------------------------------------------------------------------------

class LOWESS:
    """
    Class LOWESS (LOcally WEighted Scatterplot Smoothing).
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    

    def fit(self, X, y, sigma=0.005):
        """
        Fits the model to the data.
        
        :param X:           training data features
        :param y:           training data labels
        """
        # number of data points
        n = len(X)
        # initialize all weights    
        w = np.array([np.exp(-(X - X[i])**2 / (2 * sigma)) for i in range(n)])
        y_pred = np.zeros(n)
        
        # loop through all data points
        for i in range(n):
            weights = w[:, i]
            # calculate b matrix
            b = np.array([np.sum(weights * y), np.sum(weights * y * X)])
            # calculate A matrix
            A = np.array([
                [np.sum(weights), np.sum(weights * X)],
                [np.sum(weights * X), np.sum(weights * X * X)]
            ])
    
            theta = linalg.solve(A, b)
            y_pred[i] = theta[0] + theta[1] * X[i] 
            
        return y_pred
            

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
   
if __name__ == "__main__":
    # generate data
    X = np.linspace(0,1,100)
    y = np.sin(X * 1.5 * np.pi)
    noise = np.random.normal(loc=0, scale=0.25, size=100)
    y_noise = y + noise
    
    # perform locally weighted regression
    reg = LOWESS()
    y_pred = reg.fit(X, y_noise)
    
    # plot result
    plt.figure(figsize=(10,6))
    plt.plot(X, y_pred, color="darkblue", label="f(x)")
    plt.scatter(X, y_noise, facecolors="none", edgecolor="darkblue", label="f(x) + noise")
    plt.legend()
    plt.show()
    