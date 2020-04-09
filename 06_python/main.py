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
from utils.gif import make_gif
from utils.plotter import Plotter
from utils.data_creator import DataCreator

# classifiers
# -----------------------------------------------------------------------------
from clf.knn import kNN
from clf.svm import SVM
from clf.lda import LDA
from clf.irls import IRLS
from clf.mlp_torch import MLP
from clf.perceptron import Perceptron
from clf.decision_tree import DecisionTree
from clf.logistic_regression import LogisticRegression, LogRegOneVsOne

# unsupervised learning
# -----------------------------------------------------------------------------
from unsupervised.em import EM

# dimensionality reduction / decomposition
from unsupervised.decomposition.pca import PCA
from unsupervised.decomposition.tsne import TSNE
from unsupervised.decomposition.kernel_pca import KernelPCA
from unsupervised.decomposition.auto_encoder import AutoEncoder

# clustering
from unsupervised.clustering.k_medoids import KMedoids
from unsupervised.clustering.mean_shift import MeanShift
from unsupervised.clustering.dbscan import DBSCAN, OPTICS
from unsupervised.clustering.spectral_clustering import SpectralClustering
from unsupervised.clustering.affinity_propagation import AffinityPropagation
from unsupervised.clustering.agglomerative_clustering import AgglomerativeClustering

# regression
# -----------------------------------------------------------------------------
from reg.svr import SVR_GD, SVR_sklearn
from reg.knn_regression import KnnRegression
from reg.gaussian_process import GaussianProcess
from reg.kernel_regression import KernelRegression
from reg.bayesian_regression import BayesRegression

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
    X, y = DataCreator().make_classification(name="non_linear", n_classes=2)
    
    # train kNN classifier
#    clf = kNN(n_neighbors=1)
#    clf.fit(X, y)
    
    # train SVM classifier
#    clf = SVM(kernel="polynomial", C=1.0, p=3, s=3.0)
#    y[np.where(y == 0)] = -1
#    clf.fit(X, y)
#    clf.plot_contour(X[y == 1], X[y == -1])
    
    # train logistic regression classifier
    clf = LogisticRegression(poly=True)
    clf.fit(X, y, batch_size=X.shape[0])
    
    # train one-vs-one logistic regression classifier
#    clf = LogRegOneVsOne(poly=True)
#    clf.fit(X, y)
    
    # train an iterative reweighted least squares (IRLS) classifier
#    clf = IRLS(poly=True)
#    clf.fit(X, y, n_iter=5)
    
    # train a perceptron with rbf basis functions
#    clf = Perceptron(n_rbf_neurons=10)
#    clf.fit(X, y, sigma=1.0, alpha=0.0001, n_max_iter=1000)
#    clf.plot_contour(discrete=False)
    
    # train pytorch MLP
#    clf = MLP()
#    clf.fit(X, y, n_epochs=10000)
    
    # train Fisher's linear discriminant
#    clf = LDA(n_dims=1)
#    y[np.where(y == 0)] = -1
#    clf.fit(X, y)
    
    # train decision tree classifier
#    clf = DecisionTree()
#    clf.fit(X, y, max_depth=4, min_size=1)
#    clf.visualize()
    
    # plot boundary
    # -------------------------------------------------------------------------
    Plotter(X, y).plot_boundary(clf, step_size=0.005)
    
    # evaluation
    # -------------------------------------------------------------------------
#    evaluator = Evaluator()
#    acc = evaluator.accuracy(clf, X, y)
#    print("Accuracy: {} %".format(acc))
#    evaluator.conf_mat(clf, X, y)
#    print(evaluator.auc(clf, X, y, plot=True)) # works for logistic regression
    # print classification report
#    evaluator.classification_report(clf, X, y)
    
    
def regression():
    """
    Regression.
    """
    # create data
    X, y = DataCreator().make_regression(name="sklearn")
    
    # train Gaussian process regressor
    reg = GaussianProcess()
    reg.fit(X, y)
    reg.plot()
    
    # train kernel ridge regressor
#    reg = KernelRegression()
#    reg.fit(X, y, kernel="gaussian")
    
    # train kNN regression
#    reg = KnnRegression()
#    reg.fit(X, y, k=5)
    
    # train bayesian linear regression
#    n = 100
#    reg = BayesRegression(alpha=2.0, beta=30.0, poly=True)
#    reg.fit(X[:n, :], y[:n])
#    reg.plot(X[:n, :], y[:n], std_devs=1)
    
    # train SVR (using gradient descent)
#    reg = SVR_GD()
#    reg.fit(X, y, epsilon=0.5, n_epochs=1000, learning_rate=0.1)
    
    # train SVR (sklearn)
#    reg = SVR_sklearn()
#    reg.fit(X, y, epsilon=0.05, C=5.0, kernel="rbf")
#    reg.plot()
    
    # plot boundary
    Plotter(X, y).plot_regression(reg, n_points=500, title="kNN regression")


def unsupervised_learning():
    """
    Unsupervised learning.
    """
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import proj3d
  
    X, y = DataCreator().make_classification(name="linear", n_classes=4)
    
    # expectation maximization (EM)
    em = EM()
    em.fit(X, n_comp=3, n_iter=30)
    
    # -------------------------------------------------------------------------
    # dimensionality reduction
    # -------------------------------------------------------------------------
    # plot 3d data
#    fig = plt.figure(figsize=(8, 8))
#    ax = fig.add_subplot(111, projection="3d")
#    ax.view_init(20, -80)
#    ax.scatter(
#        X[:, 0], X[:, 1], X[:, 2], c=y, marker="o", alpha=0.5)
    
#    plt.savefig("data_viz_3d.pdf")
#    plt.show()
    
    # transform data into 2d space
    # PCA
#    pca = PCA()
#    X_hat = pca.fit_transform(X, n_components=2)
    
    # kernel PCA
#    kpca = KernelPCA()
#    X_hat = kpca.fit_transform(X, n_components=2, gamma=0.5)
    
    # t-SNE
#    tsne = TSNE()
#    X_hat = tsne.fit_transform(X, n_components=2)
    
    # auto-encoder
#    ae = AutoEncoder()
#    ae.fit(X, n_components=2, denoising=False)
#    X_hat = ae.transform()
        
    # plot 2d data
#    fig, ax = plt.subplots(figsize=(8, 8))
#    ax.scatter(
#        X_hat[:, 0], X_hat[:, 1], c=y, marker="o", alpha=0.5)
    
#    plt.savefig("data_viz_2d.pdf")
#    plt.show()
    
    # -------------------------------------------------------------------------
    # clustering
    # -------------------------------------------------------------------------
    # k-medoids clustering
#    km = KMedoids()
#    c_assign = km.fit(X, k=4)
    
    # spectral clustering
#    sc = SpectralClustering()
#    c_assign = sc.fit(X, n_clusters=2)
    
    # DBSCAN clustering
#    dbscan = DBSCAN(eps=1.00, min_pts=3)
#    c_assign = dbscan.fit(X)
    
    # OPTICS clustering
#    optics = OPTICS(eps=100.00, eps_=1.30, min_pts=3, plot_reach=True)
#    c_assign = optics.fit(X)
    
    # affinity propagation clustering
#    ap = AffinityPropagation()
#    c_assign, figs = ap.fit(X, damping=0.9, n_iter=40, plot=True)
#    make_gif(figures=figs, filename="./z_img/affinity_propagation.gif", fps=2)
    
    # mean-shift clustering
#    ms = MeanShift()
#    c_assign = ms.fit(X, bandwidth=1.0, min_dist=0.01)
    
    # agglomerative clustering
#    ac = AgglomerativeClustering()
#    c_assign = ac.fit(X, n_cluster=4, method="complete_link", dendrogram=True)
    
    # plot clusters
#    Plotter(X, y).plot_clusters(c_assign)
    
    # -1 is noise
#    print(c_assign)
    
    
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
    unsupervised_learning()
#    reinforcement_learning()
    