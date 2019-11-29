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
exploration_decay_rate = 0.001

# ========================== ALGORITHM ==========================

# store reward for all episode to see improvements
rewards_all_episodes = []


# main algorithm
for episode in range(num_episodes):
    # reset environment to starting state
    state = env.reset()

    # keep track if episode is over
    done = False

    #keep track of reward per episode
    rewards_current_episode = 0

    # single timestep
    for step in range(max_steps_per_episode):

        # Exploration vs eploitation

        # determine whether to explore or to exploit
        exploration_rate_threshold = random.uniform(0, 1)
        
        # if greater --> exploit
        if exploration_rate_threshold > exploration_rate:
            # choose the action that has the highest q value
            action = np.argmax(q_table[state,:])
        # else --> explore
        else:
            # random action
            action = env.action_space.sample()
        # do action and get information back
        new_state, reward, done, info = env.step(action)

        # Update Q-Table for Q(s, a)
        # Weighted sum of the old q value and the new q value (reward + discounted rate * maximum q value 
        # that can be achieved from any possible next state-action pair )
        # Bellman optimality equation
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state,:]))
        
        # Set state to the new state
        state = new_state
        rewards_current_episode += reward
        
        # if reached the end --> end episode
        if done == True:
            break

    # Decay exploration rate
    exploration_rate = min_exploration_rate + \
       (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    
    rewards_all_episodes.append(rewards_current_episode)

# num episode is reached
# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.array_split(np.array(rewards_all_episodes), num_episodes/1000)
count = 1000

print("\n================ Average Reward per thousand episode ================\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000
    
# Print updated Q-table
print("\n\n =========== Q-Table ===========")
print(q_table)

