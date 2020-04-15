# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:46:50 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np

from optim.particle_swarm import PSO
from optim.nelder_mead import NelderMead
from optim.simulated_annealing import SimulatedAnnealing

from utils.gif import make_gif


# -----------------------------------------------------------------------------
# Test functions
# -----------------------------------------------------------------------------
        
def func(x):
    return x[0]**2 + x[1]**2


def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def rastrigin(x):
    return 10 * 2 + (x[0]**2 - 10 * np.cos(2 * np.pi * x[0])) \
        + (x[1]**2 - 10 * np.cos(2 * np.pi * x[1]))
        
        
def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 \
        + (x[0] + x[1]**2 - 7)**2
        

def beale(x):
    return (1.5 - x[0] + x[0] * x[1])**2 \
        + (2.25 - x[0] + x[0] * x[1]**2)**2 \
        + (2.625 - x[0] + x[0] * x[1]**3)**2
        
        
def bukin(x):
    return 100 * np.sqrt(np.abs(x[1] - 0.01 * x[0]**2)) + 0.01 * np.abs(x[0] + 10)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Main entry point for optim.
    """
    # particle swarm optimization
    # -------------------------------------------------------------------------
#    pso = PSO()
#    x_min, figs = pso.optimize(
#        f=rosenbrock, n_particles=100, s_lim=(-2, 2), n_iter=200,
#        omega=0.85, phi_p=0.85, phi_g=1.00, plot=True
#    )
#    make_gif(figures=figs, filename="./z_img/particle_swarm.gif", fps=2)
    
    # nelder-mead (downhill simplex) optimization
    # -------------------------------------------------------------------------
#    nm = NelderMead()
#    x_min, figs = nm.optimize(f=rosenbrock, n_iter=20)
#    make_gif(figures=figs, filename="./z_img/nelder_mead.gif", fps=2)
    
    # simulated annealing optimization
    # -------------------------------------------------------------------------
    sa = SimulatedAnnealing()
    x_min = sa.optimize(f=rastrigin, n_iter=100, n_dim=2, s_lim=(-10, 10))
    
    print(x_min)
    