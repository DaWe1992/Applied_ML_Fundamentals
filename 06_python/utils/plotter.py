#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:37:51 2019

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# numpy and plotting
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Class Plotter
# -----------------------------------------------------------------------------

class Plotter:
    """
    Class Plotter.
    Plots the decision boundary/regression line.
    """

    def __init__(self, X, y):
        """
        Constructor.

        :param X:               data features
        :param y:               data labels
        """
        self.X = X
        self.y = y
        
        if X.shape[1] == 2: # two-dimensional data
            self.x_min, self.x_max = np.floor(self.X[:, 0].min()), \
                np.ceil(self.X[:, 0].max())
            self.y_min, self.y_max = np.floor(self.X[:, 1].min()), \
                np.ceil(self.X[:, 1].max())
        else: # one-dimensional data
            self.x_min, self.x_max = np.floor(self.X.min()), np.ceil(self.X.max())
            self.y_min, self.y_max = np.floor(self.y.min()), np.ceil(self.y.max())
                
        self.x_min -= 0.50; self.x_max += 0.50
        self.y_min -= 0.50; self.y_max += 0.50


    def __prepare_plot(self, ax, xlabel=r"$x_1$", ylabel=r"$x_2$"):
        """
        Prepares the plot.

        :param ax:              pyplot axis object
        :param xlabel:          label of the x-axis
        :param ylabel:          label of the y-axis
        """
        ax.set_xlim((self.x_min, self.x_max))
        ax.set_ylim((self.y_min, self.y_max))

        # axis labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # draw major grid
        ax.grid(b=True, which="major", color="gray", \
            linestyle="--", zorder=5)


    def plot_boundary(self, clf, step_size=0.0025):
        """
        Plots the decision boundary.

        :param clf:             classifier model
        """
        fig, ax = plt.subplots(figsize=(12.00, 7.00))
        self.__prepare_plot(ax)

        # create a mesh-grid
        xx, yy = np.meshgrid(
            np.arange(self.x_min, self.x_max, step_size),
            np.arange(self.y_min, self.y_max, step_size)
        )
        # classify each point in the mesh-grid
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # create filled contour plot
        ax.contourf(
            xx, yy, Z,
            cmap="rainbow",
            alpha=0.40, zorder=0,
            vmin=-1, vmax=np.unique(self.y).shape[0]
        )
        
        if np.unique(self.y).shape[0] == 2:
            ax.contour(xx, yy, Z, levels=[0], cmap="Greys_r", linewidths=2.5)
        
        ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y,
            cmap="rainbow", edgecolors="k", zorder=10,
            vmin=-1, vmax=np.unique(self.y).shape[0] #, s=50
        )
        
#        plt.savefig("boundary.pdf")
        plt.show()
        
        
    def plot_regression(self, reg, n_points=100):
        """
        Plots the regression line.
        
        :param reg:             regression model
        :param n_points:        number of points to be evaluated
        """
        fig, ax = plt.subplots(figsize=(12.00, 7.00))
        self.__prepare_plot(ax, xlabel=r"$x$", ylabel=r"$y$")
        
        # query query data points
        X_q = np.linspace(self.x_min - 5, self.x_max + 5, n_points).reshape(-1, 1)
        y_q = reg.predict(X_q)
        
        # draw scatter plot
        ax.plot(self.X, self.y, "rx", markersize=10, markeredgewidth=1.5)
        
        plt.plot(X_q, y_q, "m-", linewidth=2)
            
        plt.show()
