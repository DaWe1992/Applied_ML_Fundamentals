#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:06:24 2019

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# data creation and plotting
# -----------------------------------------------------------------------------
from utils.DataCreator import DataCreator
from utils.BoundaryPlotter import BoundaryPlotter

# classifiers
# -----------------------------------------------------------------------------
from clf.kNN import kNN
from clf.SVM import SVM
from clf.LogisticRegression import LogisticRegression

# regression
# -----------------------------------------------------------------------------
from reg.GaussianProcess import GaussianProcess

# evaluation
# -----------------------------------------------------------------------------
from utils.Evaluator import Evaluator


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def classification():
    """
    Classification.
    """
    # create data
    X, y = DataCreator().make_classification(sklearn=True)
    
    # train kNN classifier
    clf = kNN(n_neighbors=3)
    clf.fit(X, y)
    
    # train SVM classifier
#    clf = SVM(kernel="linear", C=1.0, p=2, s=5.0)
#    y[np.where(y == 0)] = -1
#    clf.fit(X, y)
    
    # train logistic regression classifier
#    clf = LogisticRegression()
#    clf.fit(X, y, batch_size=X.shape[0])
    
    # plot boundary
    BoundaryPlotter(X, y).plot_boundary(clf, step_size=0.005)
    
    # evaluation
    evaluator = Evaluator()
    acc = evaluator.accuracy(clf, X, y)
    print("Accuracy: {} %".format(acc))
    evaluator.conf_mat(clf, X, y)
    
    
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Main function.
    """
    regression()
    