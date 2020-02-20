#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:06:24 2019

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# numpy
# -----------------------------------------------------------------------------
import numpy as np

# data creation and plotting
# -----------------------------------------------------------------------------
from utils.data_creator import DataCreator
from utils.plotter import Plotter

# classifiers
# -----------------------------------------------------------------------------
from clf.knn import kNN
from clf.svm import SVM
from clf.lda import LDA
from clf.logistic_regression import LogisticRegression, LogRegOneVsOne
from clf.mlp_torch import MLP
from clf.decision_tree import DecisionTree

# unsupervised learning
# -----------------------------------------------------------------------------
from unsupervised.em import EM
from unsupervised.pca import PCA
from unsupervised.kernel_pca import KernelPCA
from unsupervised.tsne import TSNE
from unsupervised.auto_encoder import AutoEncoder
from unsupervised.spectral_clustering import SpectralClustering

# regression
# -----------------------------------------------------------------------------
from reg.gaussian_process import GaussianProcess
from reg.kernel_regression import KernelRegression
from reg.knn_regression import KnnRegression
from reg.bayesian_regression import BayesRegression
from reg.svr import SVR_GD, SVR_sklearn

# reinforcement learning
# -----------------------------------------------------------------------------
from rl.grid_world import GridWorld
from rl.q_learning import QLearning
from rl.value_iteration import ValueIteration
from rl.policy_iteration import PolicyIteration

# evaluation
# -----------------------------------------------------------------------------
from utils.evaluator import Evaluator


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def classification():
    """
    Classification.
    """
    # create data
    X, y = DataCreator().make_classification(name="linear", n_classes=3)
    
    # train kNN classifier
#    clf = kNN(n_neighbors=1)
#    clf.fit(X, y)
    
    # train SVM classifier
#    clf = SVM(kernel="polynomial", C=1.0, p=3, s=3.0)
#    y[np.where(y == 0)] = -1
#    clf.fit(X, y)
#    clf.plot_contour(X[y == 1], X[y == -1])
    
    # train logistic regression classifier
#    clf = LogisticRegression(poly=True)
#    clf.fit(X, y, batch_size=X.shape[0])
    
    # train one-vs-one logistic regression classifier
#    clf = LogRegOneVsOne(poly=True)
#    clf.fit(X, y)
    
    # train pytorch mlp
#    clf = MLP()
#    clf.fit(X, y)
    
    # train Fisher's linear discriminant
#    clf = LDA(n_dims=1)
#    y[np.where(y == 0)] = -1
#    clf.fit(X, y)
    
    # train decision tree classifier
#    clf = DecisionTree()
#    clf.fit(X, y, max_depth=4, min_size=1)
#    clf.visualize()
    
    # Expectation Maximization
    em = EM()
    em.fit(X, n_comp=3, n_iter=30)
    
    # plot boundary
#    Plotter(X, y).plot_boundary(clf, step_size=0.005)
    
    # evaluation
#    evaluator = Evaluator()
#    acc = evaluator.accuracy(clf, X, y)
#    print("Accuracy: {} %".format(acc))
#    evaluator.conf_mat(clf, X, y)
    
    
def regression():
    """
    Regression.
    """
    # create data
    X, y = DataCreator().make_regression(name="custom")
    
    # train Gaussian process regressor
#    reg = GaussianProcess()
#    reg.fit(X, y)
#    reg.plot()
    
    # train kernel ridge regressor
#    reg = KernelRegression()
#    reg.fit(X, y, kernel="gaussian")
    
    # train knn regression
#    reg = KnnRegression()
#    reg.fit(X, y, k=5)
    
    # train bayesian linear regression
#    n = 100
#    reg = BayesRegression(alpha=2.0, beta=30.0, poly=True)
#    reg.fit(X[:n, :], y[:n])
#    reg.plot(X[:n, :], y[:n], std_devs=1)
    
    # train svr (using gradient descent)
#    reg = SVR_GD()
#    reg.fit(X, y, epsilon=0.5, n_epochs=1000, learning_rate=0.1)
    
    # train svr (sklearn)
    reg = SVR_sklearn()
    reg.fit(X, y, epsilon=0.5, C=5.0, kernel="rbf")
    reg.plot()
    
    # plot boundary
    Plotter(X, y).plot_regression(reg, n_points=500)


def unsupervised_learning():
    """
    Unsupervised learning.
    """
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import proj3d
  
    X, y = DataCreator().make_classification(name="swiss", n_classes=2)
    
    # dimensionality reduction
    # -------------------------------------------------------------------------
    # plot 3d data
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(20, -80)
    ax.scatter(
        X[:, 0], X[:, 1], X[:, 2], c=y, marker="o", alpha=0.5)
    
#    plt.savefig("data_viz_3d.pdf")
    plt.show()
    
    # transform data into 2d space
    # pca
#    pca = PCA()
#    X_hat = pca.fit_transform(X, n_components=2)
    
    # kernel pca
#    kpca = KernelPCA()
#    X_hat = kpca.fit_transform(X, n_components=2, gamma=0.5)
    
    # t-SNE
#    tsne = TSNE()
#    X_hat = tsne.fit_transform(X, n_components=2)
    
    # auto-encoder
    ae = AutoEncoder()
    ae.fit(X, n_components=2, denoising=False)
    X_hat = ae.transform()
        
    # plot 2d data
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        X_hat[:, 0], X_hat[:, 1], c=y, marker="o", alpha=0.5)
    
#    plt.savefig("data_viz_2d.pdf")
    plt.show()
    
    # clustering
    # -------------------------------------------------------------------------
    # perform spectral clustering
#    sc = SpectralClustering()
#    c_assign = sc.fit(X, method="knn")
#    
#    fig, ax = plt.subplots(figsize=(9, 7))
#    ax.set_title("Data after spectral clustering", fontsize=18, fontweight="demi")
#    ax.scatter(X[:, 0], X[:, 1],c=c_assign ,s=50, cmap="viridis")
    
    
def reinforcement_learning():
    """
    Reinforcement learning.
    """
    # initialize environment
    # -------------------------------------------------------------------------
    # environment description
    env_description = {
        "size": {
            "x": 16,
            "y": 10
        },
        "obs_pos": [[1, 1], [3, 1], [3, 7], [0, 5], [1, 5], [5, 11], [6, 11],
                    [7, 11], [8, 11], [9, 11], [3, 11], [3, 12], [3, 13], [3, 14], [3, 15],
                    [7, 2], [6, 6], [6, 7], [7, 6], [7, 7], [5, 12], [5, 14], [4, 14],
                    [0, 8], [1, 9]],
        "r": {
            "r_p": 10,
            "r_n": -10,
            "r_o": -1,
            "r_p_pos": [[3, 3], [0, 7], [9, 10]],
            "r_n_pos": [[2, 3], [3, 6], [0, 2], [7, 3], [7, 0], [7, 1], [6, 8]]
        }
    }
    
    env = GridWorld(env_description)
    
    # initialize value iteration
    # and calculate the optimal policy
    # -------------------------------------------------------------------------
    rl = ValueIteration(gamma=0.99, thresh=10e-5, env=env)
#    rl = PolicyIteration(gamma=0.99, thresh=10e-5, env=env)
#    rl = QLearning(gamma=0.99, alpha=0.20, eps=1.00, n_episodes=5000, env=env)
    
    # get optimal policy and value function
    pi = rl.get_pi()
    V = rl.get_V()
    
    env.render()
    env.pretty_print_policy(pi, V)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Main function.
    """
#    classification()
#    regression()
#    unsupervised_learning()
    reinforcement_learning()
    