# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:39:49 2020

@author: Daniel Wehner
@see: https://github.com/akcarsten/Independent_Component_Analysis

Implementation of FastICA (Independent Component Analysis).
This algorithm can e.g. be used to decompose a mixed signal into its
source signals (BSS - blind source separation)

For a more detailed explanation of the FastICA algorithm,
see: https://en.wikipedia.org/wiki/FastICA
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

# set a seed for the random number generator for reproducibility
np.random.seed(23)


# -----------------------------------------------------------------------------
# Class ICA
# -----------------------------------------------------------------------------

class ICA:
    """
    Class ICA (Independent Component Analysis).
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    
    
    def fit(self, X):
        """
        Decomposes the signal into its components.
        
        :param X:           mixed signal to be decomposed
        :return:            decomposed signal
        """
        # center signals
        Xc, mean_x = self.__center(X)
        
        # whiten mixed signals
        Xw, _ = self.__whiten(Xc)
        
        # the whitened covariance matrix should be equal to the identity matrix
        # (approximately)
#        print(np.round(self.cov(Xw)))
        
        # get the mixing coefficients
        W = self.__fast_ica(Xw)

        # un-mix signals
        un_mixed = Xw.T @ W.T
        
        # subtract mean
        un_mixed = (un_mixed.T - mean_x).T
        
        return un_mixed
    
        
    def __fast_ica(self, X, alpha=1, thresh=1e-8, iterations=5000):
        """
        Computes the mixing coefficients to
        decompose the data into its components.
        
        :param X:           signal to be decomposed
        :param alpha:       
        :param thresh:      termination criterion
        :param iterations:  number of iterations to perform
        """
        m, n = X.shape
    
        # initialize random weights
        W = np.random.rand(m, m)
    
        # iterate over all components
        for c in range(m):
                w = W[c, :].copy().reshape(m, 1)
                # make w a unit vector
                w = w / np.sqrt((w**2).sum())
    
                i = 0
                lim = 100
                while ((lim > thresh) & (i < iterations)):
                    ws = w.T @ X
                    # pass w*s into contrast function g
                    wg = np.tanh(ws * alpha).T
                    # pass w*s into g prime
                    wg_ = (1 - np.square(np.tanh(ws))) * alpha
    
                    # update weights
                    # ---------------------------------------------------------
                    wNew = (X * wg.T).mean(axis=1) - wg_.mean() * w.squeeze()
    
                    # decorrelate weights              
                    wNew -= wNew @ W[:c].T @ W[:c]
                    wNew /= np.sqrt((wNew ** 2).sum())
    
                    # calculate limit condition
                    lim = np.abs(np.abs((wNew * w).sum()) - 1)
    
                    # Update weights
                    w = wNew
    
                    # Update counter
                    i += 1
    
                # save the weights
                W[c, :] = w.T
                
        return W


    def __center(self, X):
        """
        Subtracts the mean from the data.
        (= mean normalization)
        
        :param x:           data to be centered
        :return:            centered data
        """
        mean = np.mean(X, axis=1, keepdims=True)
        centered =  X - mean
        
        return centered, mean


    def __cov(self, X):
        """
        Computes the covariance of the input.
        
        :param X:           data to compute covariance for
        :return:            covariance matrix
        """
        mean = np.mean(X, axis=1, keepdims=True)
        m = X - mean

        return (m @ m.T) / (X.shape[1] - 1)


    def __whiten(self, X):
        """
        Transforms the observed signals in a way that potential
        correlations between the signals are removed and their
        variances equal unity.
        
        :param X:           data to be de-correlated
        :return:            whitened data
        """
        # calculate covariance matrix
        cov = self.__cov(X)
        # singular value decoposition
        U, S, V = np.linalg.svd(cov)
        # calculate diagonal matrix of eigenvalues
        d = np.diag(1.0 / np.sqrt(S))
        # calculate whitening matrix
        # the original data is projected to decorrelate it
        M = U @ d @ U.T
        # Project onto whitening matrix
        Xw = M @ X
    
        return Xw, M
    
    
# -----------------------------------------------------------------------------
# Data generation
# -----------------------------------------------------------------------------
        
def generate_mixed_signal():
    """
    Generates a mixed signal.
    
    :return:            linspace, source matrix, mixed signal
    """
    # number of samples
    ns = np.linspace(0, 200, 1000)
    
    # source matrix
    S = np.array(
        [np.sin(ns * 1),            # sine wave
        signal.sawtooth(ns * 1.9),  # sawtooth signal
        np.random.random(len(ns))   # some random signal
    ]).T
    
    # mixing matrix (this has to be approximated by the ica algorithm)
    A = np.array([
        [0.5, 1.0, 0.2],
        [1.0, 0.5, 0.4],
        [0.5, 0.8, 1.0]
    ])
    
    # mixed signal matrix
    X = (S @ A).T
    
    return ns, S, X


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
    
if __name__ == "__main__":
    """
    Main function.
    """
    # generate data
    ns, S, X = generate_mixed_signal()
    
    # plot independent source signals
    fig, ax = plt.subplots(1, 1, figsize=[18, 5])
    
    ax.plot(ns, S, lw=5)
    ax.set_xticks([])
    ax.set_yticks([-1, 1])
    ax.set_xlim(ns[0], ns[200])
    ax.tick_params(labelsize=12)
    ax.set_title("Independent sources", fontsize=25)
    
    # plot mixed signals
    fig, ax = plt.subplots(3, 1, figsize=[18, 5], sharex=True)
    
    ax[0].plot(ns, X[0], lw=5)
    ax[0].set_title("Mixed signals", fontsize=25)
    ax[0].tick_params(labelsize=12)
    
    ax[1].plot(ns, X[1], lw=5)
    ax[1].tick_params(labelsize=12)
    ax[1].set_xlim(ns[0], ns[-1])
    
    ax[2].plot(ns, X[2], lw=5)
    ax[2].tick_params(labelsize=12)
    ax[2].set_xlim(ns[0], ns[-1])
    ax[2].set_xlabel("Sample number", fontsize=20)
    ax[2].set_xlim(ns[0], ns[200])
    
    plt.show()

    ica = ICA()
    un_mixed = ica.fit(X)

    # Plot input signals (not mixed)
    fig, ax = plt.subplots(1, 1, figsize=[18, 5])
    
    ax.plot(S, lw=5)
    ax.tick_params(labelsize=12)
    ax.set_xticks([])
    ax.set_yticks([-1, 1])
    ax.set_title("Source signals", fontsize=25)
    ax.set_xlim(0, 100)
    
    fig, ax = plt.subplots(1, 1, figsize=[18, 5])
    
    ax.plot(un_mixed, "--", label="Recovered signals", lw=5)
    ax.set_xlabel("Sample number", fontsize=20)
    ax.set_title("Recovered signals", fontsize=25)
    ax.set_xlim(0, 100)
    
    plt.show()
    