# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:51:29 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal as mv_norm


# -----------------------------------------------------------------------------
# Class BayesRegression
# -----------------------------------------------------------------------------

class BayesRegression():
    """
    Class BayesRegression.     
    """
    
    def __init__(self, alpha=2.0, beta=20.0, poly=True):
        """
        Constructor.
        
        :param alpha:           precison parameter for prior distribution
        :param beta:            precision parameter for noise
        :param poly:            flag indicating whether to use
                                    polynomial basis functions
        """
        self.m = 4 if poly else 2
        self.poly = poly
        # create mean and covariance matrix
        m0 = np.zeros(self.m)
        S0 = 1 / alpha * np.identity(self.m)
    
        # set prior distribution
        self.prior = mv_norm(mean=m0, cov=S0)
        # reshape to column vector
        self.m0 = m0.reshape(m0.shape + (1,))
        self.S0 = S0
        self.beta = beta
        
        # set posterior distribution
        self.mn = self.m0
        self.Sn = self.S0
        self.posterior = self.prior
        
        
    def fit(self, X, y):
        """
        Fits the regression model to the data
        by updating the mean and the covariance of the posterior distribution.
        
        :param X:               training data (features)
        :param y:               training data (labels)
        """
        # convert labels to column vector
        y = y.reshape(y.shape + (1,))
        Phi = self.__phi(X)
        
        # update covariance matrix for posterior distribution
        self.Sn = np.linalg.inv(
            np.linalg.inv(self.S0) + self.beta * Phi.T @ Phi
        )
        
        # update mean of posterior distribution
        self.mn = self.Sn @ (
            np.linalg.inv(self.S0) @ self.m0 + self.beta * Phi.T @ y
        )
        
        self.posterior = mv_norm(mean=self.mn.flatten(), cov=self.Sn)
        
        
    def predict(self, X):
        """
        Predicts the label of unseen data.
        
        :param X:               unseen data (features)
        :return:                labels for unseen data
        """
        return (self.__phi(X) @ self.mn).squeeze()
        
        
    def __phi(self, X):
        """
        Calculates the design matrix.
        Here: Appends constant one-column.

        :param X:               original features
        :return:                augmented features (with one-column)
        """
        phi = np.ones((len(X), self.m))
        phi[:, 1] = X
        if self.poly:
            phi[:, 2] = X**2
            phi[:, 3] = X**3
        
        return phi

    
    def __std_dev(self, X):
        """
        Calculates the limit which is 'stdevs' standard deviations
        away from the mean at a given value of x.
        
        :param X:               x-axis values
        :return:                prediction limit
        """
        n = len(X)
        Phi = self.__phi(X).T.reshape((self.m, 1, n))
        
        predictions = []
        
        for idx in range(n):
            phi = Phi[:, :, idx]
            sig = 1 / self.beta + phi.T @ self.Sn @ phi
            predictions.append((np.sqrt(sig)).flatten())
            
        return np.concatenate(predictions)
    
    
    def plot_posterior(self, x1_grid, x2_grid, real_params=[]):
        """
        Generates a contour plot of the probability distribution.
        ONLY WORKS FOR POLY=FALSE!
        
        :param x1_grid:         grid of x1-values
        :param x2_grid:         grid of x2-values
        :param real_params:     real parameters of function
        """
        pos = np.empty(x1_grid.shape + (2,))
        pos[:, :, 0] = x1_grid
        pos[:, :, 1] = x2_grid
        
        fig, ax = plt.subplots(figsize=(12.00, 7.00))
        
        plt.contourf(x1_grid, x2_grid, self.posterior.pdf(pos), 20, zorder=5)
        plt.xlabel(r"$\theta_0$", fontsize=16)
        plt.ylabel(r"$\theta_1$", fontsize=16)
        
        if real_params:
            plt.scatter(real_params[0], real_params[1],
                marker="+", c="white", s=60, zorder=10)
        
#        plt.savefig("posterior.pdf")
        plt.show()
    
    
    def plot(self, X, y, real_params=None, samples=None, std_devs=None):
        """
        A helper function to plot the noisy data, the true function, 
        and optionally a set of lines sampled from the
        posterior distribution of parameters.
        
        :param X:               training data (features)
        :param y:               training data (labels)
        :param real_params:     vector of real parameters
        :param samples:         number of draws from posterior distribution
        :param std_devs:        standard deviation for limits
        """
        fig, ax = plt.subplots(figsize=(12.00, 7.00))
        # axis labels
        plt.xlabel("x")
        plt.ylabel("y")
        # axis limits
        xmin = X.min() - 0.50
        xmax = X.max() + 0.50
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((y.min() - 0.50, y.max() + 0.50))
        
        # draw major grid
        ax.grid(b=True, which="major", color="gray", \
            linestyle="--", zorder=5)
        
        # make scatter plot
        plt.scatter(X, y, alpha=0.8, edgecolors="k")
        
        # plot true function
        if real_params:
            plt.plot([xmin, xmax],
                real_function(np.array([xmin, xmax]), real_params[0], real_params[1], 0),
                "r"
            )

        # draw samples from posterior distribution and plot
        if samples:
            for weight in self.posterior.rvs(samples): 
                plt.plot([xmin, xmax],
                    real_function(np.array([xmin, xmax]), weight[0], weight[1], 0),
                    "black")
                
        # plot standard deviations
        if std_devs:
            x_range = np.linspace(X.min() - 0.50, X.max() + 0.50, 100)
            # get mean prediction
            mean = self.predict(x_range)
            # get sigma
            sigma = self.__std_dev(x_range)
            
            # plot standard deviation
            y_upper = mean + std_devs * sigma
            y_lower = mean - std_devs * sigma
            plt.fill_between(x_range, y_lower, y_upper, color="b", alpha=0.2)
            plt.plot(x_range, y_upper, "--", c="blue", linewidth=2.0)
            plt.plot(x_range, y_lower, "--", c="blue", linewidth=2.0)
            
            plt.plot(x_range, mean, c="black", linewidth=2.0)
            
#        plt.savefig("scatter.pdf")
        plt.show()
            
            
# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
        
def real_function(X, theta_0, theta_1, noise_sigma):
    """
    Evaluates the real function.
    
    :param X:           x-values
    :param theta_0:     y-intercept
    :param theta_1:     slope
    :param noise_sigma: standard deviation of noise
    :return:            y-value
    """
    n = len(X)
    if noise_sigma == 0:
        # recovers the true function
        return theta_0 + theta_1 * X
    else:
        return theta_0 + theta_1 * X + np.random.normal(0, noise_sigma, n)
    
    
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
        
if __name__ == "__main__":
    """
    Main function.
    """
    np.random.seed(20)
    
    # real function parameters
    theta_0 = -0.3
    theta_1 = 0.5
    noise_sigma = 0.2
    beta = 1 / noise_sigma**2
    
    # generate input features from uniform distribution
    # and labels
    X = np.random.uniform(-1, 1, 1000)
    y = real_function(X, theta_0, theta_1, noise_sigma)
    
    # create regressor object
    poly = False
    reg = BayesRegression(alpha=2.0, beta=beta, poly=poly)
    
    # creates a scatter plot of the data and
    # the true function
#    reg.plot(X, y, real_params=[theta_0, theta_1])
    
    # create mesh-grid
    x_grid, y_grid = np.mgrid[-1:1:0.01, -1:1:0.01]
    # plot the prior distribution (without having seen any data points)
    if not poly:
        reg.plot_posterior(x_grid, y_grid, real_params=[theta_0, theta_1])
    
    # fit model based on the first n data points
    # (updates the posterior distribution accordingly)
    n = 2
    reg.fit(X[0:n], y[0:n])
    if not poly:
        reg.plot_posterior(x_grid, y_grid, real_params=[theta_0, theta_1])
    # plot some samples from the posterior distribution
    if not poly:
        reg.plot(X[0:n], y[0:n], real_params=[theta_0, theta_1], samples=5)
    # plot mean and standard deviations
    reg.plot(X[0:n], y[0:n], real_params=[theta_0, theta_1], std_devs=1)
