# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:00:46 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.datasets import load_iris


# -----------------------------------------------------------------------------
# Train classifier and plot
# -----------------------------------------------------------------------------

X, y = load_iris(return_X_y=True)

#
# features:
#  0: sepal length in cm
#  1: sepal width in cm
#  2: petal length in cm
#  3: petal width in cm
#
# classes:
#  0: iris-setosa
#  1: iris-versicolour
#  2: iris-virginica
#

print(X)
print(y)

clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

fig = plt.figure(figsize=(20, 12))
tree.plot_tree(clf) 
