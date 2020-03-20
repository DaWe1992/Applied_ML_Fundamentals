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
    
    
    def precision(self, clf, X_test, y_test):
        """
        Computes the precision of the model.
        
        :param clf:             classifier model
        :param X_test:          test data features
        :param y_test:          test data labels
        :return:                precision
        """
        pass
    
    
    def recall(self, clf, X_test, y_test):
        """
        Computes the recall of the model.
        
        :param clf:             classifier model
        :param X_test:          test data features
        :param y_test:          test data labels
        :return:                recall
        """
        pass
    
    
    def f1_score(self, clf, X_test, y_test):
        """
        Computes the f1-score of the model.
        
        :param clf:             classifier model
        :param X_test:          test data features
        :param y_test:          test data labels
        :return:                f1-score
        """
        p = self.precision(clf, X_test, y_test)
        r = self.recall(clf, X_test, y_test)
        
        return 2 * p * r / (p + r)
    
    
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
        
        
    def auc(self, clf, X_test, y_test, plot=True):
        """
        Calculates the auc (area under the roc curve) for the trained classifier
        and plots the roc curve if specified.
        
        :param clf:             classifier model (must have a predict method
                                which supports probabilities "probs=True")
        :param X_test:          test data features
        :param y_test:          test data labels
        :param plot:            plot roc (receiver operating characteristic) curve
        :return:                auc value
        """
        # get prediction probabilities
        pred = clf.predict(X_test, probs=True)
        # concatenate predictions with true labels
        pred = np.concatenate([pred.reshape(-1, 1), y_test.reshape(-1, 1)], axis=1)
        # sort probabilities in descending order
        pred = pred[pred[:,0].argsort()[::-1]]
        
        # calculate step size for tpr and fpr
        step_n = 1 / np.where(y_test==0)[0].shape[0]
        step_p = 1 / np.where(y_test==1)[0].shape[0]
        
        # initialize roc and auc
        roc = np.zeros((X_test.shape[0] + 1, 2))
        auc = 0.00
        
        # go over all predictions
        for i in range(pred.shape[0]):
            roc[i + 1, 0] = roc[i, 0] + (step_n if pred[i, 1] == 0 else 0.00)
            roc[i + 1, 1] = roc[i, 1] + (step_p if pred[i, 1] == 1 else 0.00)
            if roc[i + 1, 0] > roc[i, 0]:
                # increase auc value
                auc += roc[i + 1, 1] * step_n
        auc = round(auc, 2)
        
        # plot roc curve
        if plot:
            self.__plot_roc(roc, auc)
        
        return auc
    
    
    def __plot_roc(self, roc, auc):
        """
        Plots the roc curve.
        
        :param roc:             array containing the roc curve
        :param auc:             auc value
        """
        fig, ax = plt.subplots()
        # plot diagonal
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        # plot roc curve
        ax.plot(
            roc[:,0], roc[:,1],
            "b", label="ROC curve",
            linewidth=2.5, markersize=5
        )
        
        # plot areas
        # aoc -- area over the curve
        ax.fill_between(roc[:,0], roc[:,1], 1, color="gray", alpha=0.2)
        # auc -- area under the curve
        ax.fill_between(roc[:,0], 0, roc[:,1], alpha=0.3)
        
        ax.text(0.825, 0.275, "AUC=" + str(auc))
        
        # axis labels
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        
        # grids
        ax.grid(b=True, which="major", color="gray", linestyle="--")
        ax.legend(loc="best")
    
#        plt.savefig("roc_auc.pdf")
        plt.show()
        
        
    def classification_report(self, clf, X_test, y_test):
        """
        Prints a classification report.
        Computes precision, recall and f1-score for each class.
        
        :param clf:             classifier model
        :param X_test:          test data features
        :param y_test:          test data labels
        """
        pass
        