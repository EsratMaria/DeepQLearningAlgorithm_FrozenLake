#!/usr/bin/env python
# coding: utf-8

# In[32]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[33]:


import numpy as np
import gym
import random

# creating the environment
env = gym.make("FrozenLake-v0")

# Defining the number of actions and states
action_size = env.action_space.n
state_size = env.observation_space.n

print(action_size)
print(state_size)

# generating the Q-table
qtable = np.zeros((state_size, action_size))
print(qtable)

# initializing the hyper parameters
episodes = 10000
lr = 0.8  # alpha
steps = 99
gamma = 0.95  # discount factor
epsilon = 1.0  # initially we keep it high because initially the environment is completely unknown to us
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

print("-------- The Q- Learning Algorithm -----------")
# we update using ==> Q(s,a) => Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]

rewards = []
env.reset()

for episode in range(episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    print("***************************************************************")
    print("Episode: {}/{}, score: {}, epsilon value: {:2}"
          .format(episode, episodes, step, epsilon))

    for step in range(steps):
        #env.render()
        # exploration -exploitation trade-off
        exp_exp_tradeoff = random.uniform(0, 1)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])
        # else we do exploration by choosing random state
        else:
            action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        qtable[state, action] = qtable[state, action] + lr *             (reward + gamma *
             np.max(qtable[new_state, :]) - qtable[state, action])

        total_rewards += reward  # update total rewards

        # Our new state is state
        state = new_state
        if done == True:
            break

    episode += 1

    # Reduce epsilon (because we need less and less exploration and more exploitation)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) *         np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

    print("Score over time: " + str(sum(rewards) / episodes))
    print(qtable)


# checking reward per 1000 episodes
rewards_per_thosand_episodes = np.split(np.array(rewards), episodes / 1000)
count = 1000
print("\n")
print("***************************************************************")
print("Average reward per thousand episodes\n")
print("***************************************************************")
for r in rewards_per_thosand_episodes:
    print(count, ": ", str(sum(r / 1000)))
    count += 1000

env.close()


# In[ ]:




