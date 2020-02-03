# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:46:50 2020

@author: Daniel Wehner
@see: https://towardsdatascience.com/t-sne-python-example-1ded9953f26
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances
from sklearn.manifold.t_sne import _joint_probabilities

from scipy import linalg
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


# -----------------------------------------------------------------------------
# Class TSNE
# -----------------------------------------------------------------------------

class TSNE:
    """
    Class TSNE.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    
    
    def fit_transform(self, X, perplexity=30, n_components=2):
        """
        Fits a TSNE model to the data.
        
        :param X:           data to be reduced in dimensionality
        """
        self.n_samples = X.shape[0]
        self.n_components = n_components
        
        # compute pairwise distances
        distances = pairwise_distances(X, metric="euclidean", squared=True)
        
        # compute joint probabilities p_ij from distances
        P = _joint_probabilities(distances=distances, \
            desired_perplexity=perplexity, verbose=False)
        
        # init low-dim embeddings with standard deviation 1e-4
        X_embedded = 1e-4 * np.random.mtrand._rand.randn(self.n_samples, n_components) \
            .astype(np.float32)
        degrees_of_freedom = max(n_components - 1, 1)
        
        return self.__tsne(P, degrees_of_freedom, X_embedded=X_embedded)


    def __tsne(self, P, degrees_of_freedom, X_embedded):
        """
        Performs optimization.
        
        :param P:                   high-dimensional affinities
        :param degrees_of_freedom:  degrees of freedom
        :param X_embedded:          randomly initialized low-dimensional embeddings
        :return:                    low-dimensional embeddings
        """
        params = X_embedded.ravel()
        # specify objective function
        obj_func = self.__kl_divergence
        # perform gradient descent
        params = self.__gradient_descent(
            obj_func, params, [P, degrees_of_freedom])
        # reshape low-dim embedding    
        X_embedded = params.reshape(self.n_samples, self.n_components)
        
        return X_embedded


    def __kl_divergence(self, params, P, degrees_of_freedom):
        """
        Computes the Kullback-Leibler (KL) divergence.
        
        :param params:              parameters to be optimized
        :param P:                   high-dimensional affinities
        :param degrees_of_freedom:  degrees of freedom
        :return:                    KL divergence, gradient
        """
        MACHINE_EPSILON = np.finfo(np.double).eps
        
        X_embedded = params.reshape(self.n_samples, self.n_components)
        
        # compute low-dimensional affinity matrix
        dist = pdist(X_embedded, "sqeuclidean") / degrees_of_freedom
        dist += 1.0
        dist **= (degrees_of_freedom + 1.0) / -2.0
        Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)
        
        # KL divergence of P and Q
        kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
        
        # gradient: dC/dY
        grad = np.ndarray((self.n_samples, self.n_components), dtype=params.dtype)
        PQd = squareform((P - Q) * dist)
        
        for i in range(self.n_samples):
            grad[i] = np.dot(np.ravel(PQd[i], order="K"), X_embedded[i] - X_embedded)
        grad = grad.ravel()
        c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
        grad *= c
        
        return kl_divergence, grad


    def __gradient_descent(self,
        obj_func, p0, args, it=0, n_iter=1000,
        patience=300,
        momentum=0.8, learning_rate=200.0, min_gain=0.01,
        min_grad_norm=1e-7
    ):
        """
        Performs the gradient descent(parameter optimization).
        
        :param obj_func:            objective function
        :param p0:                  initial parameters
        :param args:                arguments to objective function
        :param it:          
        :param n_iter:              maximum number of iterations
        :param patience:            number of iterations without improvement
        :param momentum:            momentum term
        :param learning_rate:       learning rate
        :param min_gain:            minimum improvement
        :param min_grad_norm:       minimum norm of the gradient
        :return:                    best parameters
        """
        p = p0.copy().ravel()
        update = np.zeros_like(p)
        gains = np.ones_like(p)
        error = np.finfo(np.float).max
        best_error = np.finfo(np.float).max
        best_iter = i = it
        
        # perform gradient descent steps
        for i in range(it, n_iter):
            # compute error
            error, grad = obj_func(p, *args)
            # compute gradient norm
            grad_norm = linalg.norm(grad)
            inc = update * grad < 0.0
            dec = np.invert(inc)
            gains[inc] += 0.2
            gains[dec] *= 0.8
            np.clip(gains, min_gain, np.inf, out=gains)
            grad *= gains
            # compute the update vector
            update = momentum * update - learning_rate * grad
            # update parameters
            p += update
            
            print("[t-SNE] Iteration %d: error = %.7f, gradient norm = %.7f" \
                % (i + 1, error, grad_norm))
            
            # check termination criteria
            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > patience:
                break
            if grad_norm <= min_grad_norm:
                break
        return p
    
    
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
        
#if __name__ == "__main__":
#    """
#    Main function.
#    """
#    # load data
#    X, y = load_digits(return_X_y=True)
#    
#    tsne = TSNE()
#    X_embedded = tsne.fit_transform(X)
#    
#    sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend="full")
    