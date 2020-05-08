# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:41:07 2020

@author: Daniel Wehner
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# create an artificial classification problem
X, y = make_classification(n_samples=200, n_features=2,
    n_informative=2, n_redundant=0, n_classes=2,
    n_clusters_per_class=1, class_sep=4.25, random_state=42)

clf = LogisticRegression()
clf.fit(X, y)

# create a mesh-grid
X1, X2 = np.meshgrid(
    np.linspace(min(X[:, 0]), max(X[:, 0]), 1000),
    np.linspace(min(X[:, 1]), max(X[:, 1]), 1000)
)

# classify each point in the mesh-grid
Z = clf.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)

# create a figure
fig, ax = plt.subplots(figsize=(12.00, 7.00))
# plot contour
ax.contourf(X1, X2, Z, cmap="rainbow", alpha=0.4)
ax.contour(X1, X2, Z, levels=[0], cmap="Greys_r", linewidths=2.5)
# scatter plot
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="rainbow", edgecolor="k", s=100)

# plt.savefig("contour.pdf")
plt.show()
