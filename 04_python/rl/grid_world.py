#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:29:38 2019

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import itertools
import numpy as np
import matplotlib.pyplot as plt

from enum import Enum


# -----------------------------------------------------------------------------
# Class GridWorld 
# -----------------------------------------------------------------------------

class GridWorld:
    """
    This class represents a simple grid world environment.
    """
    
    class Actions(Enum):
        """
        Enumeration of possible actions.
        """
        up = 0      # move one cell up
        down = 1    # move one cell down
        left = 2    # move one cell left
        right = 3   # move one cell right
        
    
    def __init__(self, c):
        """
        Constructor.
        
        :param c:               dictionaray containing the grid world description
            e.g.:
                {
                  "size": {                         # size of the grid
                    "x": 5,                           # size in x direction
                    "y": 5                            # size in y direction
                  },
                  "obs_pos": [[2, 2], [1, 2]],      # obstacle positions
                  "r": {                            # rewards
                    "r_p": 10,                        # positive reward (for goal state)
                    "r_n": -10,                       # negative reward (for bad state)
                    "r_o": -1,                        # rewards for all other states
                    "r_p_pos": [[5, 5]],              # goal state position
                    "r_n_pos": [[4, 4]]               # bad state position
                  }
                }
        """
        self.c = c
        self.__initialize_env()
        
        
    def reset(self):
        """
        Resets the environment to initial state.
        
        :return:                agent's initial state
        """
        return self.__initialize_env()
        
        
    def get_states(self):
        """
        Returns the indices of the possible states.
        
        :return:                list of indices of possible states
        """
        return [[y, x]
            for x in range(self.c["size"]["x"])
            for y in range(self.c["size"]["y"])
            if [y, x] not in self.c["obs_pos"]
                and [y, x] not in self.c["r"]["r_p_pos"]
                and [y, x] not in self.c["r"]["r_n_pos"]]
        
    
    def get_forbidden_state_mask(self):
        """
        Returns a mask of forbidden states.
        
        :return:                forbidden state mask
        """
        mask = np.zeros((self.c["size"]["y"], self.c["size"]["x"]))
        
        ys = range(mask.shape[1])
        xs = range(mask.shape[0])
        for (x, y) in list(itertools.product(ys, xs)):
            if [y, x] in self.c["obs_pos"] \
                or [y, x] in self.c["r"]["r_p_pos"] \
                or [y, x] in self.c["r"]["r_n_pos"]:
                mask[y, x] = 1
        
        return mask
    
    
    def get_allowed_state_mask(self):
        """
        Returns a mask of allowed states.
        
        :return:                allowed state mask
        """
        return 1 - self.get_forbidden_state_mask()
                    
        
    def get_n_states(self):
        """
        Returns the number of possible states.
        (number of cells in the grid minus obstacles)
        
        :return:                number of states
        """
        return self.c["size"]["x"] * self.c["size"]["y"] \
            - len(self.c["obs_pos"])
        
        
    def get_n_actions(self):
        """
        Returns the number of actions.
        
        :return:                number of actions
        """
        return len(self.Actions)
    
    
    def step(self, action):
        """
        Takes an action. Moves the agent's position according to
        the desired action. If there is an obstacle the agent will remain on
        its current position.
        
        :param action:          action to take, can be one of the following:
                                    Actions.up, Actions.down,
                                    Actions.left, Actions.right
        :return:                result of the step, contains the following elements:
                                    ns (next state), r (reward),
                                    done (episode done)
        """
        self.agent_state, r, done = self.__step(
            self.agent_state, action, simulate=False
        )
        
        return self.agent_state, r, done
    
    
    def simulate_step(self, state, action):
        """
        Simulates an action in a given state.
        
        :param state:           state in which action shall be tried
        :param action:          action to take in given state
        :return:                result of the step, contains the following elements:
                                    ns (next state), r (reward),
                                    done (episode done)
        """
        return self.__step(state, action, simulate=True)
    
    
    def render(self):
        """
        Renders the environment with the agent's current state.
        """
        # plot environment
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        ax.matshow(self.r, cmap=plt.cm.RdYlGn)
        
        # add labels for agent and obstacles
        for (i, j), z in np.ndenumerate(self.env):
            if z == "A" or z == "x":
                ax.text(j, i, z, ha="center", va="center", fontsize=25)
        
        ax.xaxis.set_ticks_position("bottom")
        
        # move ticks such that labels are centered
        plt.gca().set_xticks([x - 0.5 for x in plt.gca().get_xticks()][1:], minor="true")
        plt.gca().set_yticks([y - 0.5 for y in plt.gca().get_yticks()][1:], minor="true")
        plt.grid(which="minor")
        
        plt.show()
        
        
    def pretty_print_policy(self, pi):
        """
        Outputs a nice policy.
        
        :param pi:              policy
        """
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        ax.matshow(self.r, cmap=plt.cm.RdYlGn)
        
        # add labels for agent and obstacles
        for (i, j), z in np.ndenumerate(self.env):
            if z == "x": ax.text(j, i, z, ha="center", va="center", fontsize=25)
            
        # add policy arrows
        d = {0: 180, 1: 0, 2: 270, 3: 90}
        for (i, j), z in np.ndenumerate(pi):
            if z in d.keys():
                ax.text(j, i, "v", ha="center", va="center", fontsize=10, rotation=d[z])
        
        ax.xaxis.set_ticks_position("bottom")
        
        # move ticks such that labels are centered
        plt.gca().set_xticks([x - 0.5 for x in plt.gca().get_xticks()][1:], minor="true")
        plt.gca().set_yticks([y - 0.5 for y in plt.gca().get_yticks()][1:], minor="true")
        plt.grid(which="minor")
        
        plt.title("Optimal policy")
        
        plt.show()
        
     
    def __step(self, state, action, simulate=False):
        """
        Performs the actual step.
        
        :param state:           current state
        :param action:          action to take
        :param simulate:        flag indicating whether to simulate or not
        :return:                result of the step, contains the following elements:
                                    ns (next state), r (reward),
                                    done (episode done)
        """
        
        def __check_boundary(state):
            """
            Checks if the agent is still in the environment and
            has not crashed into any obstacles.
            
            :param state:       agent's current state
            :return:            true if new position is valid,
                                    false otherwise
            """
            # check if agent is still in environment boundaries
            if state[0] < 0 or state[0] >= self.c["size"]["y"] \
            or state[1] < 0 or state[1] >= self.c["size"]["x"]:
                return False
            
            # check if agent crashes into obstacle
            if state in self.c["obs_pos"]:
                return False
            
            # state is valid
            return True
            
            
        def __check_episode_done(state):
            """
            Checks if the episode is done.
            
            :param state:       agent's state
            :return:            true if the episode ends,
                                    false otherwise
            """
            # check if state is a terminal state
            if state in self.c["r"]["r_p_pos"] \
            or state in self.c["r"]["r_n_pos"]:
                return True
            
            # state is no terminal state
            return False
        
        
        ns = state[:]
        # alter position based on action
        if self.Actions(action) == self.Actions.up:
            ns[0] -= 1
        elif self.Actions(action) == self.Actions.down:
            ns[0] += 1
        elif self.Actions(action) == self.Actions.left:
            ns[1] -= 1
        else:
            ns[1] += 1
         
        # check if agent moved to a valid position
        is_valid_state = __check_boundary(ns)
        if is_valid_state:
            if not simulate:
                self.env[state[0], state[1]] = "o"
                self.env[ns[0], ns[1]] = "A"
        else: ns = state
    
        return ns, self.r[ns[0], ns[1]], __check_episode_done(ns)
            
            
    def __initialize_env(self):
        """
        Initializes the environment.
        Builds the grid world and initializes the rewards,
        sets the agent's position in the grid world.
        
        :return:                agent's initial state
        """
        size_x = self.c["size"]["x"]
        size_y = self.c["size"]["y"]
        
        # initialize environment array
        # ---------------------------------------------------------------------
        self.env = np.chararray((size_y, size_x), unicode=True)
        self.env[:] = str("o")
        # set obstacles
        for p in self.c["obs_pos"]: self.env[p[0], p[1]] = "x"
        # set goal states
        for p in self.c["r"]["r_p_pos"]: self.env[p[0], p[1]] = "g"
        # set bad states
        for p in self.c["r"]["r_n_pos"]: self.env[p[0], p[1]] = "b"
            
        # initialize reward array
        # ---------------------------------------------------------------------
        self.r = np.ones((size_y, size_x)) * self.c["r"]["r_o"]
        # set positive reward
        for p in self.c["r"]["r_p_pos"]:
            self.r[p[0], p[1]] = self.c["r"]["r_p"]
        # set negative reward
        for p in self.c["r"]["r_n_pos"]:
            self.r[p[0], p[1]] = self.c["r"]["r_n"]
        
        # set agent's position arbitrarily
        # ---------------------------------------------------------------------
        states = self.get_states()
        self.agent_state = states[np.random.choice(len(states) - 1)]
        self.env[self.agent_state[0], self.agent_state[1]] = "A"
        
        return self.agent_state
    