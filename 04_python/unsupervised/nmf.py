# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:29:03 2020

@author: Daniel Wehner
For a detailed derivation of the update rules, see:
https://www.jjburred.com/research/pdf/jjburred_nmf_updates.pdf
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np


# -----------------------------------------------------------------------------
# Class NMF
# -----------------------------------------------------------------------------

class NMF:
    """
    Class NMF.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    
    
    def fit_transform(self, X, r=2, n_max_iter=1000, eps=0.00001):
        """
        Fits the NMF model and transforms the data.
        
        :param X:           input data
        :param r:           rank parameter
        :param n_max_iter:  maximum number of iterations
        :param eps:         minimum change of matrix norm necessary
        :return:            non-negative matrix factorization W, H
        """
        self.X = X
        
        k = X.shape[0]
        n = X.shape[1]

        # initialize matrices randomly
        W = np.random.randn(k, r)
        H = np.random.randn(r, n)
        
        norm_old = np.inf
        # perform training iterations (block-coordinate descent)
        for i in range(n_max_iter):
            # update H matrix
            H *= (W.T @ X) / (W.T @ W @ H)
            # update W matrix
            W *= (X @ H.T) / (W @ H @ H.T)
            
            # calculate norm and check early stopping criterion
            norm = np.linalg.norm(X - (W @ H))
            print("Iteration {0}, norm = {1}".format(i, norm))
            
            if np.abs(norm - norm_old) < eps:
                break
            
            norm_old = norm
            
        return W, H
    
    
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
        
if __name__ == "__main__":
    X = np.asarray([
        [1, 5, 8, 1, 0, 3],
        [4, 7, 1, 0, 9, 5],
        [0, 2, 1, 1, 3, 7]        
    ])
    
    nmf = NMF()
    W, H = nmf.fit_transform(X, r=2, eps=0.00001)
    
    print(W @ H)
    