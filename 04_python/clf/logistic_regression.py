#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:55:43 2019

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# numpy and scipy
# -----------------------------------------------------------------------------
import numpy as np
#from scipy.special import binom

# itertools - get class combinations
# -----------------------------------------------------------------------------
from itertools import combinations

# tables and pretty printing
# -----------------------------------------------------------------------------
#from termcolor import colored
from prettytable import PrettyTable


# -----------------------------------------------------------------------------
# Class LogisticRegression
# -----------------------------------------------------------------------------

class LogisticRegression:
    """
    Class LogisticRegression.
    Implementation of a logistic regression classifier.
    """
    
    def fit(self,
        X, y,
        alpha=0.001,
        n_max_iter=10000,
        batch_size=1,
        epsilon=1e-15):
        """
        Fits a logistic regression model to the data.
        
        :param X:               data features (training data)
        :param y:               data labels (training data)
        :param alpha:           learning rate
        :param n_max_iter:      maximum number of iterations for gradient descent
        :param batch_size:      size of the batch used in each training iteration
        :param epsilon:         threshold used for early stopping
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
        
        print("Fitting logistic regression...")
        # fit the model parameters
        self.theta = self.__grad_desc(alpha, n_max_iter, batch_size, epsilon)
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
        :return:                theta
        """
        print("Starting gradient descent...")
        
        # initialize parameters
        theta = [0.00] * (self.m + 1)
        cost = np.inf; current_cost = np.inf
        
        # initialize a table which is going to contain the gradient descent steps
        t = PrettyTable(["Iteration", "theta_0", "theta_1", "theta_2", "J(theta)"])
        
        for i in range(n_max_iter):
            # get next batch for parameter estimation
            X_b, y_b = self.__next_batch(n=batch_size)
            cost = current_cost
            # gradient descent step
            theta -= alpha * (self.__sigmoid(X_b @ theta) - y_b) @ X_b
            # calculate cost for new theta
            current_cost = self.__cost(theta)
            # early stopping
            if np.abs(current_cost - cost) < epsilon:
                print("Early stopping... difference fell below", epsilon)
                break
            
            # print progress
            if i % 300 == 0:
                #self.plot(theta, boundary=True)
                t.add_row([
                    i,
                    "{0:.10f}".format(theta[0]),
                    "{0:.10f}".format(theta[1]),
                    "{0:.10f}".format(theta[2]),
                    "{0:.10f}".format(current_cost)
                ])
        
        print("Done.")
        print(t)
        
        return theta
    
    
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


# -----------------------------------------------------------------------------
# Class LogRegOneVsOne
# -----------------------------------------------------------------------------

class LogRegOneVsOne:
    """
    Class LogRegOneVsOne.
    Implements multi-class classification for binary classifiers.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        self.clfs = []
        self.class_map = []
        
        
    def fit(self, X, y):
        """
        Fits multiple logistic regression models to the data.
        One for each pair of classes.
        
        :param X:               data features (training data)
        :param y:               data labels (training data)
        """
        self.X = X
        self.y = y
        self.n_classes = np.unique(y).shape[0]
        combs = combinations(np.unique(y), 2)
        
#        print(colored("Fit {0} logistic regression models..." \
#            .format(binom(self.n_classes, 2)), "green"))
        
        for comb in combs:
            # prepare data
            X, y = self.__get_binary_data(comb)
            
#            print(colored("--- Learning classifier for classes {0}, {1}..." \
#                  .format(comb[0], comb[1])), "green")
            
            # learn classifier
            clf = LogisticRegression()
            clf.fit(X, y, batch_size=X.shape[0])
            self.clfs.append(clf)
            self.class_map.append((comb[0], comb[1]))
            
            
    def predict(self, X):
        """
         Predicts the labels of unseen data.
        
        :param X:               data features (test data)
        :return:                labels for test data instances
        """
        ctr = np.zeros((X.shape[0], self.n_classes))
        
        for i, clf in enumerate(self.clfs):
            # make prediction
            pred = clf.predict(X)
            # map prediction back to original classes
            mapped_pred = pred.copy()
            mapped_pred[np.where(pred == 0)] = self.class_map[i][0]
            mapped_pred[np.where(pred == 1)] = self.class_map[i][1]
        
            ctr += self.__one_hot(mapped_pred)
            
        # get final prediction
        pred = np.argmax(ctr, axis=1)
        for i in range(ctr.shape[0]):
            if np.sum(ctr[i,:] == np.amax(ctr[i,:])) > 1:
                pred[i] = -2
        
        return pred
            
            
    def __get_binary_data(self, comb):
        """
        Returns binary data.
        
        :param comb:            class combination
        :return:                binary data set with classes in combination
        """
        indices_0 = np.where(self.y == comb[0])
        indices_1 = np.where(self.y == comb[1])
        
        X = np.r_[self.X[indices_0], self.X[indices_1]]
        y = np.asarray(indices_0[0].shape[0]*[0] + indices_1[0].shape[0]*[1])

        return X, y
    
    
    def __one_hot(self, pred):
        """
        Converts predictions to one-hot encoding.
        
        :param pred:            model predictions
        :return:                one-hot encoded predictions
        """
        v = np.zeros((pred.shape[0], self.n_classes))
        v[np.arange(pred.shape[0]), pred.astype(int)] = 1
        
        return v
    