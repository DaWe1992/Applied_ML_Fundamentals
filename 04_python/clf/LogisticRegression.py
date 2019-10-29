#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:55:43 2019

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# numpy
# -----------------------------------------------------------------------------
import numpy as np

# tables
# -----------------------------------------------------------------------------
from prettytable import PrettyTable


# -----------------------------------------------------------------------------
# Class Logistic regression
# -----------------------------------------------------------------------------

class LogisticRegression:
    """
    Class LogisticRegression.
    Implementation of a logistic regression classifier.
    """
    
    def __init__(self, X, y):
        """
        Constructor.
        
        :param X:               data features (training data)
        :param y:               data labels (training data)
        """
        # determine number of training examples
        self.n = X.shape[0]
        # determine number of attributes
        self.m = X.shape[1]
        
        # add a leading 1 column (x_0 = 1)
        self.X = np.concatenate(
            (np.ones((self.n, 1)), X.reshape(-1, self.m)),
            axis=1
        )
        self.y = y
        
        # initialize parameters
        self.theta = [0.00] * (self.m + 1)
        
    
    def fit(self,
        alpha=0.001,
        n_max_iter=10000,
        batch_size=1,
        epsilon=1e-15):
        """
        Fits a logistic regression model to the data.
        
        :param alpha:           learning rate
        :param n_max_iter:      maximum number of iterations for gradient descent
        :param batch_size:      size of the batch used in each training iteration
        :param epsilon:         threshold used for early stopping
        """
        print("Fitting logistic regression...")

        # fit the model parameters
        self.__grad_desc(alpha, n_max_iter, batch_size, epsilon)
        print(self.theta)
    
    
    def predict(self, X, probs=False):
        """
        Predicts the labels of unseen data.
        
        :param X:               data features (test data)
        :param probs:           flag indicating if class labels or probabilities
                                should be returned
        :return:                labels for test data instances
        """
        # add a leading 1 column (x_0 = 1)
        X = np.concatenate(
            (np.ones((X.shape[0], 1)), X.reshape(-1, self.m)),
            axis=1        
        )
        
        pred = self.__sigmoid(X @ self.theta)
        return pred if probs else np.round(pred)
    
    
    def __grad_desc(self, alpha, n_max_iter, batch_size, epsilon):
        """
        Performs gradient descent.
        
        :param alpha:           learning rate
        :param n_max_iter:      maximum number of iterations for gradient descent
        :param batch_size:      size of the batch used in each training iteration
        :param epsilon:         threshold used for early stopping
        """
        print("Starting gradient descent...")
        
        cost = np.inf; current_cost = np.inf
        
        # initialize a table which is going to contain the gradient descent steps
        t = PrettyTable(["Iteration", "theta_0", "theta_1", "theta_2", "J(theta)"])
        
        for i in range(n_max_iter):
            # get next batch for parameter estimation
            X_b, y_b = self.__next_batch(n=batch_size)
            cost = current_cost
            # gradient descent step
            self.theta -= alpha * (self.__sigmoid(X_b @ self.theta) - y_b) @ X_b
            # calculate cost for new theta
            current_cost = self.__cost(self.theta)
            # early stopping
            if np.abs(current_cost - cost) < epsilon:
                print("Early stopping... difference fell below", epsilon)
                break
            
            # print progress
            if i % 300 == 0:
                #self.plot(theta, boundary=True)
                t.add_row([
                    i,
                    "{0:.10f}".format(self.theta[0]),
                    "{0:.10f}".format(self.theta[1]),
                    "{0:.10f}".format(self.theta[2]),
                    "{0:.10f}".format(current_cost)
                ])
        
        print("Done.")
        print(t)
    
    
    def __next_batch(self, n=1):
        """
        Returns a batch of size n.
        
        :param n:               size of the batch
        :return:                batch of size n
        """
        idx = np.arange(0 , self.n)
        np.random.shuffle(idx)
        idx = idx[:n]
        X_shuffle = [self.X[i] for i in idx]
        y_shuffle = [self.y[i] for i in idx]
    
        return np.asarray(X_shuffle), np.asarray(y_shuffle)
    
    
    def __sigmoid(self, X):
        """
        Sigmoid function.
        
        :param X:               inputs
        :return:                sigmoid function values
        """
        return 1 / (1 + np.exp(np.negative(X)))
    
    
    def __cost(self, theta):
        """
        Calculates the cost/loss for a given theta.
        
        :param theta:           model parameters
        :return:                cost associated with theta
        """
        pred = self.__sigmoid(self.X @ theta)
        
        return 1 / (2 * self.n) * sum(
            -self.y * np.log(pred)
            - (1 - self.y) * np.log(1 - pred)
        )
        