# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:46:02 2019

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import math
import numpy as np

from matplotlib import pyplot as plt


# -----------------------------------------------------------------------------
# Class EM
# -----------------------------------------------------------------------------

class EM():
    """
    Class EM.
    Implements expectation-maximization for probability density estimation.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass


    def fit(self, X, n_comp=4, n_iter=30):
        """
        Execute em (expectation-maximization) algorithm
        """
        self.X = X
        self.n = X.shape[0]
        self.m = X.shape[1]
        
        self.n_comp = n_comp
        self.n_iter = n_iter
        
        # init covariance matrices
        cov = np.array([np.identity(self.m) for _ in range(n_comp)])
        # init means
        mu = np.random.uniform( \
           np.min(self.X), np.max(self.X), \
           size=(n_comp, self.m) \
        )
        # init priors
        pi = np.random.uniform(size=(self.n_comp,))
                
        # perform em iterations
        for i in range(n_iter):
            # -----------------------------------------------------------------
            # EXPECTATION STEP
            # -----------------------------------------------------------------
            # compute responsibilities
            # (how likely is a data point to belong to a specific gaussian)
            alpha = self.__e(mu=mu, cov=cov, pi=pi)
            # -----------------------------------------------------------------
            # MAXIMIZATION STEP
            # -----------------------------------------------------------------
            # update means, covariances, priors
            mu, cov, pi = self.__m(alpha=alpha)
            
            # plot at given steps
            if i + 1 in [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30]:
                plt.figure(i)
                self.visualize(mu, cov, i)
                
            print("Finished iteration {}".format(i))
    

    def __e(self, mu, cov, pi):
        """
        Performs expectation step.
        
        :param mu:      means of the mixture components
        :param cov:     covariance matrices of the mixture components
        :param pi:      priors of the mixture components
        :return:        array of responsibilities
        """
        
        # compute responsibilities
        alpha = np.empty((self.n_comp, self.n))
        for j in range(self.n_comp):
            alpha[j] = pi[j] * self.__multivariate_gaussian(self.X, mu[j], cov[j])
    
        # sum over all responsibilities
        denominator = np.sum(alpha, axis=0)
        
        return alpha / denominator
    
    
    def __m(self, alpha):
        """
        Performs maximization step.
        
        :param alpha:   array of responsibilities
        :return:        updated parameters (mu, cov, pi)
        """
        # sum over all data points per model
        n_j = np.sum(alpha, axis=1)
    
        mu = np.zeros((self.n_comp, self.m))
        cov = np.zeros((self.n_comp, self.m, self.m))
        
        # update parameters of all gaussian components
        for j in range(self.n_comp):
            # update means
            for i, x in enumerate(self.X):
                mu[j] += (alpha[j, i] * x)
            mu[j] /= n_j[j]
    
            # update covariance
            for i, x in enumerate(self.X):
                diff = x - mu[j]
                cov[j] += alpha[j, i] * np.outer(diff, diff.T)
    
            cov[j] /= n_j[j]
    
        # update pi
        pi = n_j / self.n
    
        return mu, cov, pi


    def __multivariate_gaussian(self, X, mu, covar):
        """
        Computes the multivariate gaussian.
        
        :param X:       data
        :param mu:      mean of multivariate Gaussian
        :param covar:   covariance of multivariate Gaussian
        :return:        probability
        """
        out = np.empty(X.shape[0])
        denominator = np.sqrt((2 * math.pi) ** X.shape[1] * np.linalg.det(covar))
        covar_inv = np.linalg.inv(covar)
    
        # compute for each data point
        for i in range(X.shape[0]):
            x = X[i]
            diff = x - mu
            out[i] = np.exp(-0.5 * diff.T @ covar_inv @ diff) / denominator
        return out


    def visualize(self, mu, covar, iter):
        """
        Visualizes the em process.
        
        :param mu:      means
        :param covar:   covariances
        :param iter:    iteration counter
        """
        steps = 100
    
        x1 = self.X[:,0]
        x2 = self.X[:,1]
    
        x1_min = np.min(x1); x1_max = np.max(x1)
        x2_min = np.min(x2); x2_max = np.max(x2)
    
        x1_lin = np.linspace(x1_min - 1, x1_max + 1, steps)
        x2_lin = np.linspace(x2_min - 1, x2_max + 1, steps)
    
        Y, X = np.meshgrid(x2_lin, x1_lin)
        Z = np.empty((steps, steps))
    
        for j in range(self.n_comp):
            for i in range(steps):
                # construct vector with same x1 and all possible x2 to cover the plot space
                points = np.append(X[i], Y[i]).reshape(2, steps).T
                Z[i] = self.__multivariate_gaussian(points, mu[j], covar[j])
            plt.contour(X, Y, Z, 5)
    
        # plot the samples
        plt.plot(x1, x2, "co", zorder=1)
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.title("Mixtures after {} steps".format(iter + 1))
