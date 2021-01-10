# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 17:21:57 2018
RBFN (Radial Basis Function Network)
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

# data set creation
# -----------------------------------------------------------------------------
#from sklearn.datasets import make_blobs
#from sklearn.datasets import make_moons
#from sklearn.datasets import make_circles
from sklearn.datasets import make_classification

# classifier evaluation (confusion matrix)
# -----------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix

# data set splitting
# -----------------------------------------------------------------------------
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------------------------
# Data creation
# -----------------------------------------------------------------------------

def create_data(n_samples=150, sklearn=False):
    """
    Creates some training data.
    :return:            training data set
    """
    if sklearn:
        X, y = make_classification( \
            n_classes=2, \
            n_samples=n_samples, \
            n_features=2, \
            n_redundant=0, \
            n_informative=2, \
            n_clusters_per_class=1, \
            class_sep=0.50, \
        )
#        X, y = make_circles(n_samples=400, factor=0.30, noise=0.05)
#        X, y = make_blobs( \
#            n_samples=250, \
#            n_features=2, \
#            cluster_std=1.50, \
#            centers=2, \
#            shuffle=False \
#        )
#        X, y = make_moons(n_samples=150, noise=0.125, random_state=42)
    else:
        X = np.asarray(
            [[ 0.31740024, -0.10637588], [-0.45956105, -2.06676908],
             [-0.60394870, -1.53695769], [ 1.40859667,  1.05222070],
             [ 1.32635987,  2.01397702], [ 1.45411918,  1.22565978],
             [-1.65033002,  0.83225514], [ 0.35698076, -0.84010500],
             [ 1.08784860,  1.37498384], [ 1.30473564,  2.14172126],
             [ 1.30324092,  1.86445395], [ 1.84630365,  0.88899636],
             [ 1.73966362,  0.29913793], [ 1.68395176,  0.74838318],
             [ 1.36837616,  1.62158305], [ 1.35109333,  3.17865715],
             [ 1.61640353,  1.80140439], [-0.51032457, -0.76217301],
             [ 1.35334816,  2.77804395], [ 1.52572021,  2.08638084],
             [ 1.48135286,  1.73834622], [-3.12697332, -0.46427903],
             [-0.44432834, -2.42528327], [ 0.56288014, -1.97358039],
             [-3.64796525, -0.88910253], [-2.80278405, -0.64689998],
             [-2.44209194, -1.64000931], [-0.34758231, -1.69626078],
             [ 1.56899454,  1.27102280], [-2.28697282, -2.20312466],
             [ 1.45472301,  2.47829039], [-0.70092913,  0.40349164],
             [-1.94964412, -1.07398643], [ 0.89845739, -1.02856224],
             [-2.62420806, -0.62491013], [-0.35357207, -0.16970136],
             [ 1.90263585,  1.28858640], [-1.84619893, -1.73009037],
             [ 1.51732132,  1.55256128], [-0.62354143, -2.07630993],
             [ 1.42103289,  1.55749815], [ 1.74955203,  0.32751561],
             [-0.18995766, -1.60884076], [ 1.43974527,  2.30721871],
             [ 1.60982650,  1.68381264], [-3.34299152, -2.64985341],
             [ 1.37843602,  1.02222300], [-0.01249803, -1.91777619],
             [ 1.53860176,  0.45517841], [-0.72996989, -0.45705084],
             [ -3.5879547,  2.57439822], [-3.87452485,  2.92474585]]
        )
        y = np.asarray([
            0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 1
        ])
    
    return X, y


# -----------------------------------------------------------------------------
# Radial basis function network
# -----------------------------------------------------------------------------
    
class RBFN():
    """
    Class RBFN (radial basis function network).
    
    members:
        - X                 (training data, features)
        - y                 (training data, labels)
        - M                 (prototype vector)
        - W                 (matrix of weights)
        - sigma             (standard deviation for distance calculations)
        - alpha             (learning rate)
        - n_classes         (number of classes in the training data set)
        - n_rbf_neurons     (number of radial basis function neurons)
    """
    
    def __init__(self, n_rbf_neurons):
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
        
        # initialize prototype vectors (means)
        self.M = self.__get_prototypes()
        # initialize weight matrix (for dense layer)
        self.W = np.random.rand(
            self.M.shape[0], # + 1, # + 1 for bias term
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
#                self.plot_contour(discrete=False)
            
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
        
        
    def test(self, X_test, y_test, plot=True):
        """
        Tests the given model.
        
        :param X_test:              test data instances, features
        :param y_test:              test data instances, labels
        :param plot:                flag indicating whether to plot the confusion matrix or not
        :return:                    confusion matrix, accuracy
        """
        y_pred = self.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_pred)
        acc = np.trace(conf_mat) / np.sum(conf_mat)
        
        if plot:
            # plot confusion matrix
            fig, ax = plt.subplots()
            ax.matshow(conf_mat, cmap=plt.cm.Greys)
            
            # add labels to confusion matrix
            for (i, j), z in np.ndenumerate(conf_mat):
                ax.text(j, i, "{:0.1f}".format(z), ha="center", va="center", \
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.3") \
                )
        
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.show()
        
        return conf_mat, acc
        
        
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
#        rbf_dist = np.c_[rbf_dist, np.ones(rbf_dist.shape[0])]
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
    
    
    def plot(self):
        """
        Plots the data.
        """
        fig, ax = plt.subplots(figsize=(10.00, 5.00))
        self.__prepare_plot(ax)
        
        # draw scatter plot
        ax.scatter( \
            self.X[:,0], self.X[:,1], \
            c=self.y, cmap=plt.cm.rainbow, edgecolors="k", \
            zorder=10 \
        )
    
#        plt.savefig("data.pdf")
        plt.show()
        
        
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
        
        plt.savefig("decision_boundary.pdf")
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
    
    
# -----------------------------------------------------------------------------
# Run rbfn
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Main program.
    """
    # create data
    X, y = create_data(n_samples=150, sklearn=True)
    # train / test split
    X_train, X_test, y_train, y_test = train_test_split( \
        X, y, test_size=0.33, random_state=42 \
    )
    
    # fit radial basis function network
    # -------------------------------------------------------------------------
    rbfn = RBFN(n_rbf_neurons=20)
    rbfn.fit(X_train, y_train, sigma=1.00, alpha=0.01, n_max_iter=500)
    rbfn.plot()
    rbfn.plot_contour(discrete=False)
#    print(rbfn.predict(X, mode="raw"))
    
    # test classifier
    # -------------------------------------------------------------------------
    _, acc = rbfn.test(X_test, y_test, plot=True)
    print("Accuracy:", "{:0.2f}".format(acc))