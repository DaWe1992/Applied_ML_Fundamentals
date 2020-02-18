#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:39:58 2019

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# convex optimization
# -----------------------------------------------------------------------------
import cvxopt

# numpy
# -----------------------------------------------------------------------------
import numpy as np
import numpy.linalg as linalg

import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Class SVM
# -----------------------------------------------------------------------------
    
class SVM:
    """
    Class SVM.
    """
    
    def __init__(self, kernel="linear", C=None, p=3, s=5.0):
        """
        Constructor.
        
        :param kernel:          kernel to use (linear, polynomial, gaussian)
        :param C:               slack (soft margin)
        :param p:               degree of polynomial (only for polynomial kernel)
        :param s:               standard deviation (only for Gaussian kernel)
        """
        # choose kernel function
        if kernel == "linear":
            self.kernel = self.__linear_kernel
        elif kernel == "polynomial":
            self.kernel = self.__polynomial_kernel
        else:
            self.kernel = self.__gaussian_kernel
            
        self.p = p
        self.s = s
            
        # regularization parameter
        self.C = C
        if self.C is not None: self.C = float(self.C)


    def fit(self, X, y):
        """
        Solves the optimization problem and
        calculates the lagrange multipliers.
        
        :param X:               predictors/features
        :param y:               labels
        """
        n_samples, n_features = X.shape

        # ---------------------------------------------------------------------
        # Create matrices which are needed by 'cvxopt' package
        # ---------------------------------------------------------------------
        # gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                # apply kernel for each pair of i and j
                K[i, j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), "d")
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve qp problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # lagrange multipliers
        a = np.ravel(solution["x"])

        # support vectors have non-zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("{0} support vectors out of {1} points" \
            .format(len(self.a), n_samples))

        # intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # weight vector
        if self.kernel == self.__linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None


    def __project(self, X):
        """
        Computes y(x) without sign function.
        
        :param X:               data features (test data)
        :return:                projection
        """
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
                
            return y_predict + self.b
        
        
    def predict(self, X):
        """
        Predicts the labels of unseen data.
        
        :param X:               data features (test data)
        :return:                labels for the test data instances
        """
        return np.sign(self.__project(X))
        
        
    # -------------------------------------------------------------------------
    # Kernel functions
    # -------------------------------------------------------------------------

    def __linear_kernel(self, x1, x2):
        """
        Linear kernel. Returns the dot product of x and y.
        
        :param x1:              data point 1
        :param x2:              data point 2
        :return:                kernel value for x1 and x2
        """
        return np.dot(x1, x2)
    
    
    def __polynomial_kernel(self, x1, x2):
        """
        Polynomial kernel.
        
        :param x1:              data point 1
        :param x2:              data point 2
        :param p:               degree of the polynomial
        :return:                kernel value for x1 and x2
        """
        return (1 + np.dot(x1, x2))**self.p
    
    
    def __gaussian_kernel(self, x1, x2):
        """
        Gaussian (RBF = radial basis function) kernel.
        
        :param x1:              data point 1
        :param x2:              data point 2
        :param sigma:           standard deviation
        :return:                kernel value for x1 and x2
        """
        return np.exp(-linalg.norm(x1 - x2)**2 / (2 * (self.s**2)))
    
    
    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def plot_contour(self, X1_train, X2_train):
        """
        Plots the contours.
        
        :param X1_train:        training examples, class +1
        :param X2_train:        training examples, class -1
        """
        fig, ax = plt.subplots(figsize=(16.00, 10.00))
        plt.plot(X1_train[:,0], X1_train[:,1], "ro", zorder=10)
        plt.plot(X2_train[:,0], X2_train[:,1], "bo", zorder=10)
        plt.scatter(self.sv[:,0], self.sv[:,1], s=100, \
            c="#aaaaaa", marker="s", zorder=5)
        
        max_val = np.amax(
            np.append(
                [np.amax(X1_train, axis=0)],
                [np.amax(X2_train, axis=0)], axis=0
            ),
            axis=0
        )
        min_val = np.amin(
            np.append(
                [np.amin(X1_train, axis=0)],
                [np.amin(X2_train, axis=0)], axis=0
            ),
        axis=0)
    
        X1, X2 = np.meshgrid(
            np.linspace(
                min_val[0] - 1, max_val[0] + 1, 50
            ),
            np.linspace(
                min_val[1] - 1, max_val[1] + 1, 50
            )
        )
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = self.__project(X).reshape(X1.shape)
        plt.contour(X1, X2, Z, [0.0], colors="k", linewidths=2, origin="lower")
        plt.contour(X1, X2, Z + 1, [0.0], colors="grey",
            linewidths=1.5, linestyles="dashed", origin="lower")
        plt.contour(X1, X2, Z - 1, [0.0], colors="grey",
            linewidths=1.5, linestyles="dashed", origin="lower")
        
        # draw major grid
        ax.grid(b=True, which="major", color="lightgray", linestyle="--", zorder=-5)
    
        plt.show()
