#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:08:22 2019

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# numpy
# -----------------------------------------------------------------------------
import numpy as np

# sklearn
# -----------------------------------------------------------------------------
from sklearn.datasets import make_classification


# -----------------------------------------------------------------------------
# Class DataCreator
# -----------------------------------------------------------------------------

class DataCreator:
    """
    Class Data Creator.
    """
    
    def make_classification(self, sklearn=False):
        """
        Creates the classification data set.
        
        :param sklearn:         flag indicating if sklearn should be used
        :return:                X, y (data features and labels)
        """
        X = np.asarray(
            [[3.00, 1.00],
             [3.20, 2.20],
             [3.15, 4.80],
             [3.35, 1.20],
             [3.05, 3.50],
             [3.55, 2.85],
             [1.50, 2.25],
             [2.88, 2.18],
             [1.95, 4.00],
             [3.01, 2.95],
             [2.85, 3.01],
             [5.85, 2.20],
             [4.19, 4.00],
             [5.15, 3.50],
             [5.07, 2.89],
             [4.87, 3.54],
             [4.44, 3.78],
             [4.48, 3.94],
             [5.51, 3.80]]
        )
        y = np.asarray(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        )
        
        # sklearn data set creation
        if sklearn:
            X, y = make_classification(
                n_samples=50,
                n_features=2,
                n_redundant=0,
                n_informative=2,
                n_clusters_per_class=1,
                class_sep=1.1
            )
        
        return X, y
    