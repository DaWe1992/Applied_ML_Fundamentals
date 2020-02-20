# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:58:22 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np


# -----------------------------------------------------------------------------
# Class PolicyIteration
# -----------------------------------------------------------------------------
            
class PolicyIteration():
    """
    This class implements the policy iteration algorithm.
    This algorithm is based on the policy improvement theorem.
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
        self.pi = -np.ones((env.c["size"]["y"], env.c["size"]["x"])) \
            * self.env.get_mask()
        
        # in the beginning the policy is defined to be unstable
        policy_stable = False
        
        while not policy_stable:
            self.__policy_evaluation()
            policy_stable = self.__policy_improvement()
          
        # print results
#        print("Value function:")
#        print(self.V, "\n")
        
#        print("Optimal policy:")
#        print(self.pi, "\n")
            
            
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
        
        
    def __policy_evaluation(self):
        """
        Evaluates the current policy and computes
        the state value function.
        """
        while True:
            delta = 0
            # go over all states
            for s in self.env.get_states():
                v = self.V[s[0], s[1]]
                # unlike in value iteration we use the current policy
                # (we do not have to calculate the max over all actions)
                ns, r, _ = self.env.simulate_step( \
                    s, self.env.Actions(self.pi[s[0], s[1]]))
                self.V[s[0], s[1]] = r + self.gamma * self.V[ns[0], ns[1]]
                delta = max(delta, abs(v - self.V[s[0], s[1]]))
                
            # break loop if treshold is reached
            if delta < self.thresh:
                break
        
        
    def __policy_improvement(self):
        """
        Updates the policy based on the
        state values computed.
        """
        policy_stable = True
        # go over all states
        for s in self.env.get_states():
            s_old = self.pi[s[0], s[1]]
            q = np.zeros(self.env.get_n_actions())
           
            # go over all actions
            for i in range(q.shape[0]):
                # get i-th action
                a = self.env.Actions(i)
                # try i-th action in state s
                ns, r, _ = self.env.simulate_step(s, a)
                q[i] = r + self.gamma * self.V[ns[0], ns[1]]
                
            # get the argmax => best action in state s
            self.pi[s[0], s[1]] = np.argmax(q)
            # check if the policy has changed
            # if it has => policy is unstable => further iterations
            # if not => we can stop!
            if s_old != self.pi[s[0], s[1]]:
                policy_stable = False
                
        return policy_stable
    