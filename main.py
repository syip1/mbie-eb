#!/usr/bin/env python3
import gym
import numpy as np
from q_learning import Q_Learning
from mbie_eb import MBIE_EB

environment = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True)
'''
rewards = []
for i in range(100, 3000, 100):
    print('Training for', i, 'episodes.')
    learn = Q_Learning(environment)
    learn.train(n_episodes=i, silent=True)
    reward = learn.test()
    rewards.append(reward)
print(rewards)
'''
learn = Q_Learning(environment)
learn.train(10000)
learn.test()
'''
learn2 = MBIE_EB(environment)
learn2.train(10000)
'''
