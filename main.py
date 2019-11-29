import numpy as np
import gym
import random
import time
from IPython.display import clear_output

# Set up environment (query info, sample states/action, retrieve reward)
env = gym.make("FrozenLake-v0")

# Determine the number of actions
action_space_size = env.action_space.n

# Determine the number of states
state_space_size = env.observation_space.n

#creat a q-table (initialize to all 0s)
q_table = np.zeros((state_space_size, action_space_size))

"""
[0, 0, 0, 0...]
[0, 0, 0, 0...]
[0, 0, 0, 0...]
[., ., ., ....]
[., ., ., ....]
[., ., ., ....]
"""

# define number of episodes to play
num_episodes = 10000

# define the max number of steps before an episode ends
max_steps_per_episode = 100

# define the learning rate
learning_rate = 0.1

# define the discount rate
discount_rate = 0.99

# exploration vs exploitation
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01