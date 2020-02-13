# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:36:31 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sklearn.svm import SVR


# -----------------------------------------------------------------------------
# Class SVR_sklearn
# -----------------------------------------------------------------------------

class SVR_sklearn:
    """
    Class SVR_sklearn.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    
    
    def fit(self, X, y, epsilon=0.5, C=1.0, kernel="rbf"):
        """
        Fits an SVR model to the data using gradient descent.
        
        :param X:               training data (features)
        :param y:               training data (labels)
        :param epsilon:         parameter for the epsilon-tube without penalty
        :param C:               regularization parameter
        :param kernel:          kernel function
        """
        self.X = X
        self.y = y
        self.epsilon = epsilon
        
        self.svr = SVR(epsilon=epsilon, C=C, kernel=kernel, gamma="auto")
        self.svr.fit(X, y)
    
    
    def predict(self, X):
        """
        Predicts the label of unseen data.
        
        :param X:               unseen data (features)
        :return:                labels of unseen data
        """
        return self.svr.predict(X)
    
    
    def plot(self):
        """
        Plots the svr result along with the epsilon-tube.
        """
        fig, ax = plt.subplots(figsize=(12.00, 7.00))
        # axis labels
        plt.xlabel("x")
        plt.ylabel("y")
        # axis limits
        xmin = self.X.min() - 0.50
        xmax = self.X.max() + 0.50
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((self.y.min() - 0.50, self.y.max() + 0.50))
        
        # draw major grid
        ax.grid(b=True, which="major", color="gray", \
            linestyle="--", zorder=5)
        
        # make scatter plot
        plt.scatter(self.X, self.y, alpha=0.8, edgecolors="k")
        
        # make predictions and plot prediction
        x_range = np.linspace(self.X.min() - 0.50, self.X.max() + 0.50, 100)
        y_pred = self.predict(x_range.reshape(-1, 1))
        plt.plot(x_range, y_pred, c="black", linewidth=2.0)
        
        # plot epsilon-tube
        plt.fill_between(x_range, y_pred - self.epsilon, y_pred + self.epsilon, color="b", alpha=0.2)
        plt.plot(x_range, y_pred + self.epsilon, "--", c="blue", linewidth=1.0)
        plt.plot(x_range, y_pred - self.epsilon, "--", c="blue", linewidth=1.0)
        
        plt.show()
    

# -----------------------------------------------------------------------------
# Class SVR_GD (Support Vector Regression) using gradient descent
# -----------------------------------------------------------------------------

class SVR_GD:
    """
    Class SVR.
    Supports only linear regression (without kernel)
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
        
        
    def fit(self, X, y, epsilon=0.5, n_epochs=100, learning_rate=0.1):
        """
        Fits an SVR model to the data using gradient descent.
        
        :param X:               training data (features)
        :param y:               training data (labels)
        :param epsilon:         parameter for epsilon band
        :param n_epochs:        number of epochs for training  
        :param learning_rate:   learning rate
        """
        # initialize a tensorflow session
        self.sess = tf.Session()
        
        m = X.shape[-1] if len(X.shape) > 1 else 1
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, m))
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        
        self.W = tf.Variable(tf.random_normal(shape=(m, 1)))
        self.b = tf.Variable(tf.random_normal(shape=(1,)))
        
        self.y_pred = tf.matmul(self.X, self.W) + self.b
        
        # define loss function
        # if the error is lower than epsilon => no cost
        self.loss = tf.norm(self.W) / 2 + tf.reduce_mean(
            tf.maximum(0.0, tf.abs(self.y_pred - self.y) - epsilon))
        
        # initialize gradient descent object
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        opt_op = opt.minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())
        
        # perform training
        # ---------------------------------------------------------------------
        for i in range(n_epochs):
            loss = self.sess.run(
                self.loss, {
                    self.X: X,
                    self.y: y
                })
            print("{} / {}: loss: {}".format(i + 1, n_epochs, loss))
            
            # optimize parameters
            self.sess.run(
                opt_op, {
                    self.X: X,
                    self.y: y
                })
            
            
    def predict(self, X):
        """
        Predicts the label of unseen data.
        
        :param X:               unseen data (features)
        :return:                labels of unseen data
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        y_pred = self.sess.run(
            self.y_pred, {
                self.X: X 
            })
        
        return y_pred
    