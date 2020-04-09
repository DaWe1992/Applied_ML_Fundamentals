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

from tqdm import tqdm


# -----------------------------------------------------------------------------
# Class PSO
# -----------------------------------------------------------------------------

class PSO():
    """
    Class PSO.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    
             
    def optimize(self, f,
        n_particles=10,
        n_dim=2,
        s_lim=(-10, 10),
        n_iter=1000,
        omega=0.85,
        phi_p=0.85,
        phi_g=0.90,
        plot=True
    ):
        """
        Optimizes function f.
        
        :param f:               function to be optimized
        :param n_particles:     number of particles
        :param n_dim:           number of dimensions
        :param s_lim:           search space limit
        :param n_iter:          number of iterations
        :param plot:            flag indicating whether to plot the process
        :return:                optimum (minimum), list of figures
        """
        figs = []
        # initialize the swarm
        swarm = np.random.uniform(
            low=s_lim[0], high=s_lim[1], size=(n_particles, n_dim))
        # vectorize function f
        f = np.vectorize(f, signature="(n)->()")
        # compute function value for all particles
        f_swarm = f(swarm)
        # best positions of particles
        p = np.copy(swarm)
        # check if a new minimum has been found
        g = swarm[np.argmin(f_swarm, axis=0)]
        # initialize the particles' velocity
        v = np.random.uniform(
            low=-abs(s_lim[1] - s_lim[0]), high=abs(s_lim[1] - s_lim[0]),
            size=swarm.shape)
        
        # ---------------------------------------------------------------------
        # optimization
        # ---------------------------------------------------------------------
        for k in tqdm(range(n_iter)):
            r_p = np.random.uniform(0, 1, swarm.shape)
            r_g = np.random.uniform(0, 1, swarm.shape)
            # compute velocity vector
            v = omega * v + phi_p * r_p * (p - swarm) \
                + phi_g * r_g * (g - swarm)
                
            # update particles' position
            swarm += v
            
            # update the paricles' best known position
            # -----------------------------------------------------------------
            ind = np.where(f(swarm) < f(p))
            p[ind] = swarm[ind]
            
            # update the swarm's best known position
            # -----------------------------------------------------------------
            ind = np.argmin(f(p), axis=0)
            if f(p[ind]) < f(g):
                g = p[ind]
             
            if plot:
                if n_dim != 2:
                    raise ValueError(
                        "Wrong dimensionality for plotting. Set n_dim=2")
                elif k % 10 == 0:
                    figs.append(self.__plot(f, swarm, s_lim))

        return g, figs
            
            
    def __plot(self, f, swarm, s_lim):
        """
        Plots the optimization progress.
        
        :param f:               function to be optimized
        :param swarm:           particle positions
        :param s_lim:           search space limit
        :return:                figure
        """
        x1, x2 = np.meshgrid(
            np.linspace(s_lim[0], s_lim[1], 300),
            np.linspace(s_lim[0], s_lim[1], 300)
        )
        
        fig, ax = plt.subplots(figsize=(10.0, 10.0))
        ax.set_xlim((s_lim[0], s_lim[1]))
        ax.set_ylim((s_lim[0], s_lim[1]))

        # create contour plot
        cf = ax.contourf(
            x1, x2, f(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape[0],-1),
            levels=30, zorder=5)
        c = ax.contour(cf, colors="k", zorder=5)
#        ax.clabel(c, c.levels, inline=True, fontsize=10)
        # plot particles
        ax.scatter(swarm[:,0], swarm[:,1], c="w", edgecolors="k", s=75, zorder=10)
        
        plt.title("Particle swarm optimization", fontsize=18, fontweight="demi")
        ax.set_xlabel(r"$x_1$", fontsize=18)
        ax.set_ylabel(r"$x_2$", fontsize=18)
        
        plt.show()
        
        return fig
        