#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:34:43 2019

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np


# -----------------------------------------------------------------------------
# Class ValueIteration
# -----------------------------------------------------------------------------
            
class ValueIteration():
    """
    This class implements the value iteration algorithm.
    """
    
    def __init__(self, gamma, thresh, env):
        """
        Constructor.
        
        :param gamma:           discount factor
        :param thresh:          threshold indicating if state values have converged
        :param env:             grid world environment
        """
        self.env = env
        self.gamma = gamma
        self.thresh = thresh
        self.V = np.zeros((env.c["size"]["y"], env.c["size"]["x"]))
        self.pi = -np.ones((env.c["size"]["y"], env.c["size"]["x"]))
        
        # calculate value function
        self.__calculate_state_values()
        print("Value function:")
        print(self.V, "\n")
        
        # calculate optimal policy
        print("Optimal policy:")
        print(self.pi, "\n")
        
        
    def get_pi(self):
        """
        Gets the policy.
        """
        return self.pi
    
    
    def get_V(self):
        """
        Gets the value function.
        """
        return self.V
    
    
    def __calculate_state_values(self):
        """
        Calculates all the state values.
        """
        while True:
            delta = 0
            # go over all states
            for s in self.env.get_states():
                v = self.V[s[0], s[1]]
                q = np.zeros(self.env.get_n_actions())
                
                # go over all actions
                for i in range(q.shape[0]):
                    # get i-th action
                    a = self.env.Actions(i)
                    # try i-th action in state s
                    ns, r, _ = self.env.simulate_step(s, a)
                    q[i] = r + self.gamma * self.V[ns[0], ns[1]]
                    
                # get the max
                self.V[s[0], s[1]] = np.max(q)
                self.pi[s[0], s[1]] = np.argmax(q)
                delta = max(delta, abs(v - self.V[s[0], s[1]]))
                
            # break loop if treshold is reached
            if delta < self.thresh:
                break
            