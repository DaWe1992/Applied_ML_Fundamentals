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
# Class BoundaryPlotter
# -----------------------------------------------------------------------------

class BoundaryPlotter:
    """
    Class BoundaryPlotter.
    Plots the decision boundary.
    """

    def __init__(self, X, y):
        """
        Constructor.

        :param X:               data features
        :param y:               data labels
        """
        self.X = X
        self.y = y
        self.x_min, self.x_max = np.floor(self.X[:, 0].min()) - 0.50, \
            np.ceil(self.X[:, 0].max()) + 0.50
        self.y_min, self.y_max = np.floor(self.X[:, 1].min()) - 0.50, \
            np.ceil(self.X[:, 1].max()) + 0.50


    def __prepare_plot(self, ax):
        """
        Prepares the plot.

        :param ax:              pyplot axis object
        """
        ax.set_xlim((self.x_min, self.x_max))
        ax.set_ylim((self.y_min, self.y_max))

        # axis labels
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        # draw major grid
        ax.grid(b=True, which="major", color="gray", linestyle="--", zorder=5)


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
            vmin=-1, vmax=np.unique(self.y).shape[0]
        )
        
#        plt.savefig("boundary.pdf")
        plt.show()
