#!/bin/bash/env python3
from discrete_env import Discrete_Env
import numpy as np

class MBIE_EB(Discrete_Env):
    """Implements the MBIE_EB algorithm on tabular MDPs.
    """
    def __init__(self, env, epsilon=0.01, gamma=0.9, delta=0.01):
        """Initialises objects to be used during learning.
        Args:
            env (gym.Env): OpenAI Gym environment.
            gamma (float): discount factor.
            delta (float): probability for confidence intervals.
        """
        super().__init__(env, gamma) 
        self.epsilon = epsilon
        self.delta = delta
        #Optimistic initialisation
        self.Q = np.ones(shape=(self.S, self.A), dtype=np.float32) / (1 - self.gamma)
        self.R = np.zeros(shape=(self.S, self.A))
        #Uniform transition probabilities
        self.T = np.ones(shape=(self.S, self.A, self.S), dtype=np.float32) / self.S
        self.n = np.ones(shape=(self.S, self.A), dtype=np.float32)

    def run_one_episode(self, tol=1e-3):
        """Trains the model on one episode.
        Returns:
            rewards (float): total reward for that episode.
        """
        state = self.env.reset()
        state = self.state_to_ind(state)
        rewards = 0
        while True:
            a = np.argmax(self.Q[state, :])
            next_state, reward, done, _ = self.env.step(a)
            next_state = self.state_to_ind(next_state)
            rewards += reward
            #Update arrays
            self.n[state, a] += 1
            self.R[state, a] += 1.0 / self.n[state, a] * (reward - self.R[state, a])
            self.T[state, a, :] += 1.0 / self.n[state, a] * ((np.arange(self.S) == next_state) - self.T[state, a, :])
            state = next_state
            #Recognise terminal states
            if (done):
                self.T[state, :, :] = np.zeros(shape=(self.A, self.S))
                break
      
        return rewards

    
    def train(self, n_episodes=1000, silent=False):
        """Trains the MBIE-EB network.
        Args:
            n_episodes (int): number of episodes to train on.
            silent (bool): whether to print training results or not.
        """
        reward_arr = []
        for i in range(n_episodes):
            print(i)
            rewards = self.run_one_episode()
            reward_arr.append(rewards)
            if i % 100 == 99 and not silent:
                #Print summary
                print('Episode', i-99, '-', i)
                print('Reward:', round(np.mean(np.array(reward_arr)), 3), '+/-', round(np.std(np.array(reward_arr)), 3))
                reward_arr = []
