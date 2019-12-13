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
from utils.boundary_plotter import BoundaryPlotter

# classifiers
# -----------------------------------------------------------------------------
from clf.knn import kNN
from clf.svm import SVM
from clf.lda import LDA
from clf.logistic_regression import LogisticRegression, LogRegOneVsOne

# unsupervised learning
# -----------------------------------------------------------------------------
from unsupervised.em import EM

# regression
# -----------------------------------------------------------------------------
from reg.gaussian_process import GaussianProcess

# reinforcement learning
# -----------------------------------------------------------------------------
from rl.grid_world import GridWorld
from rl.value_iteration import ValueIteration

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
    X, y = DataCreator().make_classification(sklearn=True, n_classes=3)
    
    # train kNN classifier
#    clf = kNN(n_neighbors=3)
#    clf.fit(X, y)
    
    # train SVM classifier
#    clf = SVM(kernel="polynomial", C=1.0, p=2, s=5.0)
#    y[np.where(y == 0)] = -1
#    clf.fit(X, y)
    
    # train logistic regression classifier
#    clf = LogisticRegression(poly=False)
#    clf.fit(X, y, batch_size=X.shape[0])
    
    # train one-vs-one logistic regression classifier
    clf = LogRegOneVsOne(poly=True)
    clf.fit(X, y)
    
    # train Fisher's linear discriminant
#    clf = LDA(n_dims=1)
#    y[np.where(y == 0)] = -1
#    clf.fit(X, y)
    
    # Expectation Maximization
#    em = EM()
#    em.fit(X, n_comp=3, n_iter=30)
    
    # plot boundary
    BoundaryPlotter(X, y).plot_boundary(clf, step_size=0.005)
    
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
    X, y = DataCreator().make_regression(sklearn=False)
    
    # train Gaussian process regressor
    gp = GaussianProcess()
    gp.fit(X, y)
    gp.plot()
    
    
def reinforcement_learning():
    """
    Reinforcement learning.
    """
    # initialize environment
    # -------------------------------------------------------------------------
    # environment description
    env_description = {
        "size": {
            "x": 8,
            "y": 4
        },
        "obs_pos": [[1, 1], [3, 1], [3, 7], [0, 5], [1, 5]],
        "r": {
            "r_p": 10,
            "r_n": -10,
            "r_o": -1,
            "r_p_pos": [[3, 3], [0, 7]],
            "r_n_pos": [[2, 3], [3, 6], [0, 2]]
        }
    }
    
    env = GridWorld(env_description)
    
    # initialize value iteration
    # and calculate the optimal policy
    # -------------------------------------------------------------------------
    val_it = ValueIteration(gamma=0.99, thresh=10e-5, env=env)
    # get optimal policy
    pi = val_it.get_pi()
    env.pretty_print_policy(pi)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Main function.
    """
#    classification()
    regression()
#    reinforcement_learning()
    