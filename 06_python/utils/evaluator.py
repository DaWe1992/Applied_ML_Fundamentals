#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:42:40 2019

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# numpy and plotting
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# classifier evaluation
# -----------------------------------------------------------------------------
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# -----------------------------------------------------------------------------
# Class Evaluator
# -----------------------------------------------------------------------------

class Evaluator:
    """
    Class Evaluator.
    """
    
    def accuracy(self, clf, X_test, y_test):
        """
        Computes the accuracy of the model.
        
        :param clf:             classifier model
        :param X_test:          test data features
        :param y_test:          test data labels
        :return:                accuracy
        """
        return accuracy_score(y_test, clf.predict(X_test)) * 100
    
    
    def conf_mat(self, clf, X_test, y_test):
        """
        Computes and plots the confusion matrix.
        
        :param clf:             classifier model
        :param X_test:          test data features
        :param y_test:          test data labels
        """
        y_pred = clf.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_pred)
        
        # plot confusion matrix
        fig, ax = plt.subplots()
        ax.matshow(conf_mat, cmap=plt.cm.Blues)
        
        # add labels to confusion matrix
        for (i, j), z in np.ndenumerate(conf_mat):
            ax.text(j, i, "{:0.1f}".format(z), ha="center", va="center",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.3")
            )
    
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Gold label")
        
        plt.show()
        