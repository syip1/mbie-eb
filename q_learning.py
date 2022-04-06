#!/usr/bin/env python3
import gym
import numpy as np
from discrete_env import Discrete_Env

class Q_Learning(Discrete_Env):
    """Implements Q learning for tabular MDPs.
    """
    def __init__(self, env, gamma=1, alpha=0.1):
        """Initialises objects to be used during learning.
        Args:
            env (gym.Env): OpenAI Gym environment. Must have discrete action and observation spaces.
            gamma (float): discount factor.
            alpha (float): learning rate.
        """
        super().__init__(env, gamma)
        self.Q = np.zeros((self.S, self.A))
        self.epsilon = 1
        self.alpha = alpha

    def run_one_episode(self):
        """Trains the model on one episode.
        Returns:
            rewards (float): total reward for that episode.
        """

        state = self.env.reset()
        state = self.state_to_ind(state)
        rewards = 0
        while True:
            die = np.random.uniform()
            if (die < self.epsilon):
                a = self.env.action_space.sample()
            else:
                a = np.argmax(self.Q[state])
            new_state, reward, done, _ = self.env.step(a)
            new_state = self.state_to_ind(new_state)
            update = reward + self.gamma * np.max(self.Q[new_state]) - self.Q[state, a]
            self.Q[state, a] += self.alpha * update
            rewards += reward
            state = new_state
            if (done):
                #for i in range(self.A):
                #    self.Q[state, i] += self.alpha * reward
                break
        return rewards


    def train(self, n_episodes=1000, silent=False):
        """Trains the Q network completely.
        Args:
            n_episodes (int): number of episodes to train on.
            silent (bool): whether or not to print training results.
        """
        reward_arr = []
        for i in range(n_episodes):
            self.epsilon = max(1 - 0.95*i/n_episodes, 0)
            self.alpha *= np.exp(np.log(0.1)/n_episodes)
            rewards = self.run_one_episode()
            reward_arr.append(rewards)
            #Adjust epsilon and alpha
            if i % 100 == 99 and not silent:
                #Print summary
                print('Episode', i-99, '-', i)
                print('Reward:', round(np.mean(np.array(reward_arr)), 3), '+/-',round(np.std(np.array(reward_arr)), 3))
                reward_arr = []
        print(self.Q)


    def test(self, n_episodes=100):
        """Tests the environment with the current greedy policy.
        Args:
            n_episodes (int): number of episodes to run on.
        """
        total_rewards = []
        for i in range(n_episodes):
            state = self.env.reset()
            state = self.state_to_ind(state)
            rewards = 0
            while True:
                a = np.argmax(self.Q[state])
                state, reward, done, _ = self.env.step(a)
                state = self.state_to_ind(state)
                rewards += reward
                if (done):
                    total_rewards.append(rewards)
                    break
        print('Reward:', np.mean(np.array(total_rewards)), '+/-', np.std(np.array(total_rewards)))
        return np.mean(np.array(total_rewards))
