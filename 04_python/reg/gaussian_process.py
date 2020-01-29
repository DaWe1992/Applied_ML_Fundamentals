#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 18:33:53 2019

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# numpy and plotting
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt 

# scipy
# -----------------------------------------------------------------------------
from scipy.spatial.distance import cdist


# -----------------------------------------------------------------------------
# Class GaussianProcess
# -----------------------------------------------------------------------------

class GaussianProcess:
    """
    Class GaussianProcess.
    """
    
    def __init__(
        self,
        sigma_n=0.01,
        sigma_f=1.00,
        length_scale=1.00
    ):
        """
        Constructor.
        
        :param sigma_n:         sigma noise (regularization parameter)
        :param sigma_f:         first hyperparameter of rbf-kernel
        :param length_scale:    second hyperparameter of rbf-kernel
        """
        self.sigma_n = sigma_n
        self.sigma_f = sigma_f
        self.length_scale = length_scale
    
    
    def __calculate_cov(self, X_1, X_2):
        """
        Calculates the covariance matrix of X and Y.
    
        :param X:       data set
        :param Y:       data set
        :return:        covariance matrix
        """        
        pairwise_sq_dists = cdist(X_1.reshape(-1, 1), X_2.reshape(-1, 1))**2
        
        return self.sigma_f * np.exp(-pairwise_sq_dists / (2 * self.length_scale)**2)

            
    def fit(self, X, y):
        """
        Fits the model to the data.
        
        :param X:       training data (features)
        :param y:       training data (labels)
        """
        print("Fitting model...")
        self.X = X
        self.y = y
        self.n_data = self.X.shape[0]
        
        # initialize kernel matrix
        self.K = np.zeros((self.n_data, self.n_data))
        self.K = self.__calculate_cov(X, X) + self.sigma_n * np.eye(self.n_data)
        self.K_inv = np.linalg.solve(self.K, np.eye(self.n_data))
        
        print("Finished fitting model.")
        
        
    def predict(self, X_q):
        """
        Predics the labels of new data points.
        
        :param X_q:     query data points
        :return:        mean vector
        """
        K_s = self.__calculate_cov(self.X, X_q)
        K_ss = self.__calculate_cov(X_q, X_q)
        
        # matrix of regression coefficients / calculate mean
        mu = self.y.T @ self.K_inv @ K_s
        # Schur complement / calculate covariance
        self.sigma = K_ss - (self.K_inv @ K_s).T @ K_s
        
        return mu
    
    
    def plot(self):
        """
        Plot the regression line.
        """
        x_from = np.min(self.X) - 0.50
        x_to = np.max(self.X) + 0.50
    
        y_from = np.min(self.y) - 0.50
        y_to = np.max(self.y) + 0.50
        X_q = np.linspace(x_from, x_to, (((x_to - x_from) / 0.05) + 1).astype(int))
        
        # predict mu and sigma
        mu = self.predict(X_q)
    
        # calculate 99 % / 95 % / 90 % confidence intervals
        var = np.diagonal(self.sigma)
        sigma_90 = 1.65 * np.sqrt(var)
        sigma_95 = 1.96 * np.sqrt(var)
        sigma_99 = 2.58 * np.sqrt(var)
        
        # -------------------------------------------------------------------------
        # plot regression line (mean of posterior distribution)
        # (and 99 % / 95 % / 90 % confidence intervals)
        # (and some other samples from the distribution)
        # -------------------------------------------------------------------------
        
        fig, ax = plt.subplots(figsize=(15.00, 5.00))
        
        # plot confidence intervals
        ax.fill_between(X_q, mu - sigma_99, mu + sigma_99, facecolor="gray", alpha=0.1)
        ax.fill_between(X_q, mu - sigma_95, mu + sigma_95, facecolor="gray", alpha=0.2)
        ax.fill_between(X_q, mu - sigma_90, mu + sigma_90, facecolor="gray", alpha=0.3)
        
        # sample some other possible functions
        k = 5
        samples = np.random.multivariate_normal(mu, self.sigma, k)
        for i in range(k):
            ax.plot(X_q, samples[i], "k-", linewidth=0.5)
        
        # plot data and mean of distribution (regression line)
        ax.scatter(self.X, self.y, s=70)
        ax.plot(X_q, mu, linewidth=2.5)
        
        ax.set_xlim((x_from, x_to))
        ax.set_ylim((y_from, y_to))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        ax.grid(b=True, which="major", color="gray", linestyle="--")
        
        plt.show()
