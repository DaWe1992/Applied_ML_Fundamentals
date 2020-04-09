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
from mpl_toolkits.mplot3d import Axes3D  


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
        
        :param X:       data
        :param n_comp:  number of Gaussian components
        :param n_iter:  number of iterations
        """
        self.X = X
        self.n = X.shape[0]
        self.m = X.shape[1]
        
        self.n_comp = n_comp
        self.n_iter = n_iter
        
        # init covariance matrices
        self.cov = np.array([np.identity(self.m) for _ in range(n_comp)])
        # init means
        self.mu = np.random.uniform( \
           np.min(self.X) + 1, np.max(self.X) - 1, \
           size=(n_comp, self.m) \
        )
        # init priors
        self.pi = np.random.uniform(size=(self.n_comp,))
                
        # perform em iterations
        for i in range(n_iter):
            # -----------------------------------------------------------------
            # EXPECTATION STEP
            # -----------------------------------------------------------------
            # compute responsibilities
            # (how likely is a data point to belong to a specific gaussian)
            alpha = self.__e()
            # -----------------------------------------------------------------
            # MAXIMIZATION STEP
            # -----------------------------------------------------------------
            # update means, covariances, priors
            self.__m(alpha=alpha)
            
            # plot at given steps
            if i + 1 in [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30]:
                plt.figure(i, figsize=(10, 7))
                self.__visualize(i)
                
            print("Finished iteration {}".format(i + 1))
        self.__visualize_surface()
        
            
    def predict(self, X):
        """
        Predicts the label of new instances.
        
        :param X:       unseen data
        :return:        labels
        """
        return np.argmax(
            self.__compute_responsibilities(X), axis=0)
    

    def __e(self):
        """
        Performs expectation step.
        
        :return:        array of responsibilities
        """
        return self.__compute_responsibilities(self.X)
        
    
    def __compute_responsibilities(self, X):
        """
        Computes the responsibilities for each Gaussian component.
        
        :param X:       data
        :return:        array of responsibilities
        """
        # compute responsibilities
        alpha = np.empty((self.n_comp, X.shape[0]))
        for j in range(self.n_comp):
            alpha[j] = self.pi[j] * self.__multivariate_gaussian(
                X, self.mu[j], self.cov[j])
    
        # sum over all responsibilities
        denominator = np.sum(alpha, axis=0)
        
        return alpha / denominator
        
    
    def __m(self, alpha):
        """
        Performs maximization step.
        
        :param alpha:   array of responsibilities
        """
        # sum over all data points per model
        n_j = np.sum(alpha, axis=1)
        # update parameters of all gaussian components
        for j in range(self.n_comp):
            # update means
            for i, x in enumerate(self.X):
                self.mu[j] += (alpha[j, i] * x)
            self.mu[j] /= n_j[j]
    
            # update covariance
            for i, x in enumerate(self.X):
                diff = x - self.mu[j]
                self.cov[j] += alpha[j, i] * np.outer(diff, diff.T)
    
            self.cov[j] /= n_j[j]
    
        # update pi
        self.pi = n_j / self.n
        

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


    def __visualize(self, iter):
        """
        Visualizes the em process.
        
        :param iter:    iteration counter
        """
        steps = 100
        X, Y = self.__get_mesh(steps)
        Z = np.empty((steps, steps))
        
        # plot 2d gaussians
        for j in range(self.n_comp):
            for i in range(steps):
                points = np.append(X[i], Y[i]).reshape(2, steps).T
                Z[i] = self.__multivariate_gaussian(points, self.mu[j], self.cov[j])
            plt.contour(X, Y, Z, 10, zorder=15, cmap="Greys")
    
        # plot the samples
        plt.scatter(self.X[:,0], self.X[:,1], zorder=10, cmap="rainbow",
            edgecolors="k", c=self.predict(self.X))
        # grid lines
        plt.grid(b=True, which="major", color="gray", linestyle="--", zorder=5)
        # labels and title
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.title("Mixtures after {} steps".format(iter + 1))
        
        
    def __visualize_surface(self):
        """
        Plots a 3D surface plot.
        """
        steps = 100
        X, Y = self.__get_mesh(steps)
        Z = np.zeros((steps, steps))
        
        # plot 3d gaussian mixture density
        for j in range(self.n_comp):
            for i in range(steps):
                points = np.append(X[i], Y[i]).reshape(2, steps).T
                Z[i] += self.__multivariate_gaussian(points, self.mu[j], self.cov[j])
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection="3d")
        
        ax.view_init(45, 35)
        ax.set_xlabel(r"$x_1$", fontsize=24)
        ax.set_ylabel(r"$x_2$", fontsize=24)
        
        surf = ax.plot_surface(X, Y, Z,
            rstride=1, cstride=1, cmap="coolwarm", edgecolor="none")
        fig.colorbar(surf, shrink=0.5, aspect=5)
#        ax.scatter(self.X[:,0], self.X[:,1], np.asarray([0.1]*len(self.X)))
        
        plt.savefig("./z_img/em.png")
        plt.show()
        
        
    def __get_mesh(self, steps):
        """
        Gets the meshgrid.
        
        :return:                meshgrid
        """
        x1 = self.X[:,0]
        x2 = self.X[:,1]
    
        x1_min = np.min(x1); x1_max = np.max(x1)
        x2_min = np.min(x2); x2_max = np.max(x2)
    
        x1_lin = np.linspace(x1_min - 1, x1_max + 1, steps)
        x2_lin = np.linspace(x2_min - 1, x2_max + 1, steps)
    
        Y, X = np.meshgrid(x2_lin, x1_lin)

        return X, Y        
