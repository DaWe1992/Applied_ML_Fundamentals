# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 17:21:57 2018
Perceptron with radial basis functions

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# numpy and plotting
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# clustering for prototype positioning and distance calculations
# -----------------------------------------------------------------------------
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# -----------------------------------------------------------------------------
# Class Perceptron
# -----------------------------------------------------------------------------
    
class Perceptron():
    """
    Class Perceptron.
    """
    
    def __init__(self, n_rbf_neurons=10):
        """
        Constructor.
        
        :param n_rbf_neurons:       number of radial basis function neurons
        """
        self.n_rbf_neurons = n_rbf_neurons
    
    
    def fit(self,
        X, y,
        sigma=1.0,
        alpha=0.0001,
        n_max_iter=10000):
        """
        Fits the rbfn model.
        
        :param X:                   training data, features
        :param y:                   training data, labels
        :param sigma:               standard deviation for distance metric
        :param alpha:               learning rate for gradient descent
        :param n_max_iter:          maximum number of iterations of gradient descent
        """
        self.X = X
        self.y = y
        
        # determine number of classes in the training data set
        self.n_classes = np.unique(y).shape[0]
        # create one-hot encoding for the labels
        self.y_one_hot = self.__one_hot(self.y)
        
        self.sigma = sigma
        self.alpha = alpha
        
        self.x_min, self.x_max = np.floor(self.X[:, 0].min()) - 0.50, \
            np.ceil(self.X[:, 0].max()) + 0.50
        self.y_min, self.y_max = np.floor(self.X[:, 1].min()) - 0.50, \
            np.ceil(self.X[:, 1].max()) + 0.50
        
        # initialize prototype vectors (means of rbfs)
        self.M = self.__get_prototypes()
        # initialize weight matrix (for dense layer)
        self.W = np.random.rand(
            self.M.shape[0] + 1, # + 1 for bias term
            self.n_classes
        )
        
        # start training
        self.__train(n_max_iter)
        
    
    def __train(self, n_max_iter):
        """
        Trains the network by updating the weights.
        The update is guided by a stochastic gradient descent.
        
        :param n_max_iter:          number of max. iterations of stochastic
                                    gradient descent
        """
        print("Starting training...")
        print("Iter", "\t", "J()")
        
        # perform n_max_iter iterations
        for i in range(1, n_max_iter + 1):
            # print progress
            if i % 100 == 0:
                print("{:06d}".format(i), "\t", "{:015.8f}".format(self.__J()))
            
            for i in range(self.X.shape[0]):
                # calculate activations (forward pass)
                # ---------------------------------------------------------
                act, dist = self.__forward_pass(np.asarray([self.X[i, :]]))
                # update weights using a single example
                # stochastic gradient descent
                # ---------------------------------------------------------
                self.__update_weights(self.y_one_hot[i, :], act, dist)
        
        print("Finished training.")
#        print(self.W)
        
            
    def predict(self, X, mode="argmax"):
        """
        Predicts the labels of unseen data.
        
        :param X:                   data instances to predict labels for
        :param mode:                indicates what the output is
                                        argmax => labels
                                        raw    => all activations
        :return:                    predictions
        """
        act = self.__forward_pass(X)[0]
        # return arg max class label or raw activations if specified
        if mode == "argmax":
            return np.argmax(act, axis=1)
        elif mode == "raw":
            return act
        
        
    def __forward_pass(self, X):
        """
        Performs a forward pass through the network.
        
        :param X:                   network input to calculate activations for
        """
        # calculate rbf distances
        # -----------------------------------------
        pw_sq_d = cdist(X, self.M)**2
        rbf_dist = np.exp(-pw_sq_d / (2 * self.sigma))
        
        # concatenate bias column
        rbf_dist = np.c_[rbf_dist, np.ones(rbf_dist.shape[0])]
        # calculate output activations
        # -----------------------------------------
        pre_act = rbf_dist @ self.W
        # apply softmax activation
        act = np.exp(pre_act) / np.sum(np.exp(pre_act), axis=1).reshape(-1, 1)
        return act, rbf_dist
    
    
    def __update_weights(self, y, act, dist):
        """
        Updates weights of the network.
        
        :param y:                   true label
        :param act:                 network activations
        :param dist:                distances to prototypes (= input to dense layer)
        """
        # stochastic gradient descent
        err_grad = (-dist).T @ (y - act)
        self.W -= self.alpha * err_grad
    
    
    def __get_prototypes(self):
        """
        Gets the prototypes using k-means clustering
        """
        # perform k-means clustering to get the prototypes
        k_means = KMeans(n_clusters=self.n_rbf_neurons).fit(self.X)
        return k_means.cluster_centers_
        
        
    def __one_hot(self, y):
        """
        Converts y-labels to one-hot-coding.
        
        :param y:                   training data, labels
        :return:                    training data, labels (one-hot)
        """
        n_data = y.shape[0]
        
        one_hot = np.zeros((n_data, self.n_classes))
        one_hot[np.arange(n_data), y] = 1
        
        return one_hot


    def __J(self):
        """
        Calculates the loss for the current model.
        
        :return:                    loss of current model
        """
        return sum(
            (self.y_one_hot.reshape(-1, 1) - \
             self.predict(self.X, mode="raw").reshape(-1, 1))**2
        )[0]
    
    
    def plot_contour(self, discrete=True):
        """
        Plots the decision boundary.
        
        :param discrete:            Indicates if plot should be discrete (labels)
                                    or continuous (probabilities).
                                    The latter only works for two classes.
        """
        fig, ax = plt.subplots(figsize=(12.00, 7.00))
        self.__prepare_plot(ax)
        
        # create a mesh-grid
        xx, yy = np.meshgrid( \
            np.arange(self.x_min, self.x_max, 0.005), \
            np.arange(self.y_min, self.y_max, 0.005) \
        )
        # classify each point in the mesh-grid
        act = self.predict(np.c_[xx.ravel(), yy.ravel()], mode="raw")
        Z = np.argmax(act, axis=1)
        
        # plot contour based on label only
        if not discrete:
            max_act = np.max(act, axis=1)
            Z[np.where(Z==1)] = -1
            Z[np.where(Z==0)] = 1   
            Z = Z * max_act

        Z = Z.reshape(xx.shape)
        # create filled contour plot
        cf = ax.contourf( \
            xx, yy, Z, levels=20, \
            cmap=plt.cm.rainbow if discrete else plt.cm.RdBu, \
            alpha=0.40, zorder=0 \
        )
        
        if not discrete:
            # plot dashed contour lines
            ax.contour(xx, yy, Z, levels=20, linestyles="dashed", \
                cmap=plt.cm.RdBu, zorder=0)
            # plot decision boundary (= contour 0)
            ax.contour(xx, yy, Z, levels=[0], cmap="Greys_r", linewidths=2.5)
            fig.colorbar(cf, shrink=0.8, extend="both")
            
        # draw data scatter plot
        ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, \
               cmap=plt.cm.rainbow, edgecolors="k", zorder=10, label="Data")
        # plot prototypes
        ax.plot(self.M[:, 0], self.M[:, 1], "w*", \
            markersize=12, markeredgecolor="k", zorder=10, label="Prototypes")
        ax.legend(loc="best")
        
        plt.show()
        
        
    def __prepare_plot(self, ax):
        """
        Prepares the plot.
        
        :param ax:                  pyplot axis object
        """
        ax.set_xlim((self.x_min, self.x_max))
        ax.set_ylim((self.y_min, self.y_max))
        
        # axis labels
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        # draw major grid
        ax.grid(b=True, which="major", color="gray", linestyle="--", zorder=5)
    