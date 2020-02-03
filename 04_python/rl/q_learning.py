# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:56:29 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np


# -----------------------------------------------------------------------------
# Class QLearning
# -----------------------------------------------------------------------------
            
class QLearning():
    """
    This class implements the Q-learning algorithm (temporal difference learning).
    This is an OFF-policy TD control algorithm.
    """
    
    def __init__(self, gamma, alpha, eps, n_episodes, env):
        """
        Constructor.
        
        :param gamma:           discount factor
        :param alpha:           learning rate
        :param eps:             epsilon (epsilon-greedy)
        :param n_episodes:      number of episodes
        :param env:             grid world environment
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.n_episodes = n_episodes
        
        self.Q = np.random.rand(env.c["size"]["y"], env.c["size"]["x"], env.get_n_actions())
        self.pi = -np.ones((env.c["size"]["y"], env.c["size"]["x"]))
        
        # calculate the Q values
        self.__calcualte_q_values()
        # calculate the policy
        self.__calculate_policy()
        
        
    def get_pi(self):
        """
        Gets the optimal policy.
        
        :return:                optimal policy
        """
        return self.pi
    
        
    def __calculate_policy(self):
        """
        Calculates the optimal policy.
        """
        for s in self.env.get_states():
            self.pi[s[0], s[1]] = np.argmax(self.Q[s[0], s[1]])
        
        
    def __calcualte_q_values(self):
        """
        Calculates the Q values for all state action pairs.
        """
        # play several episodes
        for e in range(self.n_episodes):
#            print("Episode: ", e)
            # re-initialize environment, put agent in arbitrary starting state
            s = self.env.reset()
            done = False
            
            # choose actions while the episode has not ended
            while not done:
                # choose a from s using policy derived from Q
                a = np.argmax(self.Q[s[0], s[1]]) \
                    if np.random.rand(1)[0] > self.eps \
                    else np.random.randint(0, self.env.get_n_actions())
                # take action a, observe r and s'
                ns, r, done = self.env.step(a)
                # update Q-values
                # Q(s, a) = Q(s, a) + alpha * (r + gamma * max_a'(Q(s', a')) - Q(s, a))
                self.Q[s[0], s[1], a] += self.alpha * ( \
                      r + self.gamma * np.max(self.Q[ns[0], ns[1]]) - self.Q[s[0], s[1], a] \
                )
                
                # set state to next state
                s = ns
            
            # anneal epsilon
            self.eps *= 0.99