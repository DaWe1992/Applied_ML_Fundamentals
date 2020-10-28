# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:18:08 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np


# -----------------------------------------------------------------------------
# Class PCA
# -----------------------------------------------------------------------------

class PCA():
    """
    Class PCA.
    Implements Principal Component Analysis.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    
    
    def fit_transform(self, X, n_components):
        """
        Reduces the dimensionality of the data.
        
        :param X:               high dimensional data
        :param n_components:    number of components
        :return:                data set reduced in dimensionality
        """
        # compute scatter / covariance matrix
        sigma = np.cov(X, rowvar=False)

        # compute Eigenvectors and Eigenvalues and visualize
        eig_val, eig_vec = np.linalg.eig(sigma)

        # sort eigenvectors by decreasing eigenvalues
        idx = eig_val.argsort()[::-1]   
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:,idx]
        
        U = eig_vec[:, :n_components]
        
        # transform the samples onto the new subspace
        return X @ U
    
    
## -----------------------------------------------------------------------------
## Plot eigenvectors
## -----------------------------------------------------------------------------
#
#class Arrow3D(FancyArrowPatch):
#    def __init__(self, xs, ys, zs, *args, **kwargs):
#        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
#        self._verts3d = xs, ys, zs
#
#    def draw(self, renderer):
#        xs3d, ys3d, zs3d = self._verts3d
#        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
#        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
#        FancyArrowPatch.draw(self, renderer)
#
#fig = plt.figure(figsize=(7, 7))
#ax = fig.add_subplot(111, projection="3d")
#
#ax.plot(
#    data_set[0,:], data_set[1,:], data_set[2,:], "o",
#    markersize=8, color="green", alpha=0.2
#)
#ax.plot(
#    [mean[0][0]], [mean[1][0]], [mean[2][0]], "o",
#    markersize=10, color="red", alpha=0.5
#)
#
#for v in eig_vec_cov.T:
#    a = Arrow3D(
#        [mean[0][0], v[0]], [mean[1][0], v[1]], [mean[2][0], v[2]],
#        mutation_scale=20, lw=3, arrowstyle="-|>", color="r"
#    )
#    ax.add_artist(a)
#    
#ax.set_xlabel("x_values")
#ax.set_ylabel("y_values")
#ax.set_zlabel("z_values")
#
#plt.title("Eigenvectors")
#
#plt.show()
