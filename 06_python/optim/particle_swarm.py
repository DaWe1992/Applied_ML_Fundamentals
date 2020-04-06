# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:54:57 2020
Particle Swarm Optimization (PSO)

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Class PSO
# -----------------------------------------------------------------------------

class PSO():
    """
    Class PSO.
    """
    
    def __init__(self, n_particles=10, n_dim=2, b_lo=-10, b_up=10):
        """
        Constructor.
        
        :param n_particles:     number of particles
        :param n_dim:           number of dimensions
        :param b_lo:            lower boundary of the search space
        :param b_up:            upper boundary of the search space
        """
        self.n_particles = n_particles
        self.b_lo = b_lo
        self.b_up = b_up
        # initialize the swarm
        self.swarm = np.random.uniform(low=b_lo, high=b_up, size=(n_particles, n_dim))
        self.g = None
        
             
    def optimize(self, f, n_iter, omega=0.85, phi_p=0.85, phi_g=1.00):
        """
        Optimizes function f.
        
        :param f:               function to be optimized
        :param n_iter:          number of iterations
        :return:                optimum
        """
        # vectorize function f
        f = np.vectorize(func, signature="(n)->()")
        # compute function value for all particles
        f_swarm = f(self.swarm)
        # best positions of particles
        self.p = np.copy(self.swarm)
        # check if a new minimum has been found
        self.g = self.swarm[np.argmin(f_swarm, axis=0)]
        # initialize the particles' velocity
        v = np.random.uniform(low=-abs(self.b_up - self.b_lo), high=abs(self.b_up - self.b_lo), size=self.swarm.shape)
        
        for k in range(n_iter):
            # print(i)
            print(f(self.g))
            # for each particle
            for i in range(self.swarm.shape[0]):
                # print(self.g)
                # for each dimension
                for j in range(self.swarm.shape[1]):
                    # pick random numbers
                    r_p = np.random.uniform(0, 1)
                    r_g = np.random.uniform(0, 1)
                    # update the particles' velocity
                    v[i, j] = omega * v[i, j] + phi_p * r_p * (self.p[i, j] - self.swarm[i, j]) \
                        + phi_g * r_g * (self.g[j] - self.swarm[i, j])
                
                # update the particle's position
                self.swarm[i,:] += v[i,:]
                
                # update the particle's best known position
                if f(self.swarm[i,:]) < f(self.p[i,:]):
                    self.p[i,:] = self.swarm[i,:]
                    
                    # update the swarm's best known position
                    if f(self.p[i,:]) < f(self.g):
                        self.g = self.p[i,:]
                        
            if k % 2 == 0:
                self.__plot(f)
                        
        print(self.g)
        
        
    def __plot(self, f):
        """
        
        """
        t1, t2 = np.meshgrid(
            np.linspace(-10, 10, 300),
            np.linspace(-10, 10, 300)
        )
        
        fig, ax = plt.subplots(figsize=(10.0, 10.0))
        ax.set_xlim((-10, 10))
        ax.set_ylim((-10, 10))
        # levels for contour plot
        # levels = np.asarray([0.005, 0.020, 0.080, 0.300, 0.500, 1.000, 2.000])
        # create contour plot
        c = ax.contourf(t1, t2, f(np.c_[t1.ravel(), t2.ravel()]).reshape(t1.shape[0],-1)) #, levels)
        # ax.clabel(c, c.levels, inline=True, fontsize=10)
        ax.scatter(self.swarm[:,0], self.swarm[:,1])
        
        # plt.title("Gradient descent on cost function")
        # ax.set_xlabel(r"$\theta_0$")
        # ax.set_ylabel(r"$\theta_1$")
        # ax.set_ylim((-0.25, 0.75))
        
        plt.show()
        
                        
            
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
        
def func(x):
    """
    Function to be optimized.
    """
    return x[0]**2 + x[1]**2
        

if __name__ == "__main__":
    pso = PSO(n_particles=100, b_lo=-10, b_up=10)
    pso.optimize(f=func, n_iter=1000)