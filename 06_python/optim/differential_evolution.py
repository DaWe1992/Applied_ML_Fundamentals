# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:48:54 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Differential evolution
# -----------------------------------------------------------------------------

def de(f, bounds, mut=0.8, cross_prob=0.7, pop_size=20, n_iter=1000):
    """
    Differential evolution.
    
    :param f:                   function to be optimized
    :param bound:               search range
    :param mut:                 mutation range
    :param cross_prob:          cross probability
    :param pop_size:            size of the population
    :n_iter:                    number of iterations
    """
    dimensions = len(bounds)
    pop = np.random.rand(pop_size, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    # denormalize population
    pop_denorm = min_b + pop * diff
    
    # compute initial fitness
    fitness = np.asarray([f(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    
    # perform evolution steps
    for i in range(n_iter):
        # for each individual in the population
        for j in range(pop_size):
            # choose three random individuals for each individual
            idxs = [idx for idx in range(pop_size) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < cross_prob
            
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
                
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            fitness_trial = f(trial_denorm)
            
            # replace with trial individual if it is better
            if fitness_trial < fitness[j]:
                fitness[j] = fitness_trial
                pop[j] = trial
                
                if fitness_trial < fitness[best_idx]:
                    best_idx = j
                    
        yield min_b + pop * diff, fitness, best_idx
        

# -----------------------------------------------------------------------------
# Model and cost function
# -----------------------------------------------------------------------------

def model_func(x, w):
    """
    Model function.
    """
    return w[0] + w[1] * x + w[2] * x**2 + \
        w[3] * x**3 + w[4] * x**4 + w[5] * x**5


def rmse(w):
    """
    Root mean squared error function.
    """
    y_pred = model_func(X, w)
    return np.sqrt(sum((y - y_pred)**2) / len(y))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # generate data    
    X = np.linspace(0, 10, 500)
    y = np.cos(X) + np.random.normal(0, 0.2, 500)

    # compute results
    result = list(de(rmse, [(-5, 5)] * 6, n_iter=1500))
    
    # plot results
    for i in range(0, len(result), 20):
        fig, ax = plt.subplots(figsize=(12.00, 7.00))
        ax.set_xlim([0, 10])
        ax.set_ylim([-2, 2])
        ax.scatter(X, y, c="red", edgecolor="k", alpha=0.7)
        
        ax.set_title("Regression with Differential Evolution", fontsize=18)
        # axis labels
        ax.set_xlabel("$x$", fontsize=18)
        ax.set_ylabel("$y$", fontsize=18)
        
        # draw major grid
        ax.grid(b=True, which="major", color="lightgray", zorder=5)
            
        pop, fit, idx = result[i]
        
        for ind in pop:
            data = model_func(X, ind)
            ax.plot(X, data, "k", alpha=0.7)
            
        plt.show()
        plt.close(fig)
