# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 13:41:53 2019
Policy Gradient Algorithm: REINFORCE
@see: https://www.datahubbs.com/policy-gradients-with-reinforce/
@see: https://www.datahubbs.com/reinforce-with-pytorch/
@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch import optim


# -----------------------------------------------------------------------------
# Neural network representing the policy
# -----------------------------------------------------------------------------

class policy_estimator():

    def __init__(self, env):
        """
        Constructor.
        
        :param env:             environment
        """
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        
        # network definition
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16), 
            nn.ReLU(), 
            nn.Linear(16, self.n_outputs),
            nn.Softmax(dim=-1)
        )
    
    
    def predict(self, state):
        """
        Predicts next action.
        
        :param state:           current state
        :return:                probability distribution over actions
        """
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs
    
    
def discount_rewards(rewards, gamma=0.99):
    """
    Discounts the rewards.
    
    :param rewards:             undiscounted rewards
    :param gamma:               discount factor
    :return:                    discounted rewards
    """
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean() # r - baseline (reduces variance)


def reinforce(
    env,
    policy_estimator,
    num_episodes=2000,
    batch_size=10,
    gamma=0.99):
    """
    REINFORCE implementation.
    
    :param env:                 environment
    :param policy_estimator:    neural network policy
    :param num_episodes:        number of episodes
    :param batch_size:          batch size
    :param gamma:               discount factor
    :return:                    total rewards
    """

    # set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1
    
    # define optimizer
    optimizer = optim.Adam(
        policy_estimator.network.parameters(), 
        lr=0.01 # learning rate
    )
    
    action_space = np.arange(env.action_space.n)
    
    # go over epsisodes
    for ep in range(num_episodes):
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        done = False
        
        # while the episodes has not finished
        while not done:
            # get the probability distribution over possible actions
            action_probs = policy_estimator.predict(s_0).detach().numpy()
            # select an action according to the probabilities
            action = np.random.choice(action_space, p=action_probs)
            # execute the action
            s_1, r, done, _ = env.step(action)
            
            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1
            
            # we have reached a terminal state
            if done:
                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))
                
                # update network as soon as we have
                # collected enough data (batch is full)
                if batch_counter == batch_size:
                    optimizer.zero_grad() # reset gradient to zero
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(batch_rewards)
                    # actions are used as indices, must be LongTensor
                    action_tensor = torch.LongTensor(batch_actions)
                    
                    # calculate loss
                    logprob = torch.log(
                        policy_estimator.predict(state_tensor)
                    )
                    selected_logprobs = reward_tensor * \
                        logprob[np.arange(len(action_tensor)), action_tensor]
                    loss = -selected_logprobs.mean()
                    
                    # calculate gradients
                    loss.backward()
                    # apply gradients
                    optimizer.step()
                    
                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1
                    
                # print running average
                print("\rEp: {} Average of last 10: {:.2f}".format(
                    ep + 1, np.mean(total_rewards[-10:])), end="")
                
    return total_rewards
    
    
# create environment
env = gym.make("CartPole-v0")
s = env.reset()
pe = policy_estimator(env)

# TRAINING
# -----------------------------------------------------------------------------
# apply the algorithm
rewards = reinforce(env, pe, num_episodes=10000)
window = 10
smoothed_rewards = [
    np.mean(rewards[i-window:i+1])
        if i > window 
        else np.mean(rewards[:i+1])
            for i in range(len(rewards))]

plt.figure(figsize=(12,8))
plt.plot(rewards)
plt.plot(smoothed_rewards)
plt.ylabel("Total Rewards")
plt.xlabel("Episodes")
plt.show()

# Render sample episode
# -----------------------------------------------------------------------------
done = False
sum_reward = 0.00
s = env.reset()

while not done:
    action_probs = pe.predict(s).detach().numpy()
    # select an action according to the probabilities
    action = np.argmax(action_probs)
    # execute the action
    s, r, done, _ = env.step(action)
    sum_reward += r
    # render environment
    env.render()
    
print("Accumulated reward: ", sum_reward)