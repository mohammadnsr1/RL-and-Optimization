
import tensorflow as tf
import numpy as np 
import gym
import random
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import display, clear_output
# Parameters
num_episodes = 200
learning_rate = 0.5
discount_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.001
exploration_decay_rate = 0.05
timestep = 0
num_runs = 100
# Random generator
rand_generator = np.random.RandomState(0)


# Environment
env = gym.make("CliffWalking-v0")
# env = gym.make("CliffWalking-v0",render_mode='human')
action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.zeros((state_space_size, action_space_size))
all_reward_sum = []

def update():
    env.render()
    plt.figure(figsize=(5,5))
    display(plt.gcf())
    clear_output(wait=True)


def ArgMax(q_values):
    top = float('-inf')
    ties = []
    for i in range(len(q_values)):
        if q_values[i] > top:
            top = q_values[i]
            ties = []
        if q_values[i] == top:
            ties.append(i)
    return rand_generator.choice(ties)

for run in tqdm(range(num_runs)):
    exploration_rate = 1
    episode_rewards = []
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        state = state[0]
        terminal = False
        rewards_current_episode = 0
        while not terminal:
            timestep += 1
            # exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
            exploration_rate = 0.1
            current_q = q_table[state, :]
            if rand_generator.rand() < exploration_rate:
                action = rand_generator.randint(action_space_size)
            else:
                action = ArgMax(current_q)
            
            new_state, reward, terminal, _, _ = env.step(action)
            q_table[state, action] += learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]) - q_table[state, action])
            
            state = new_state
            rewards_current_episode += reward
            if terminal:
                break

        episode_rewards.append(rewards_current_episode)
    
    all_reward_sum.append(episode_rewards)

mean_rewards = np.mean(all_reward_sum, axis=0)
plt.plot(mean_rewards)
plt.title("Rewards per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.ylim(-100,0)
plt.xlim(0,250)
plt.show()
env.close()

env = gym.make("CliffWalking-v0",render_mode='human')
state = env.reset()[0]
done = False
while not done:
    action = ArgMax(q_table[state])
    state, _, done, _,_ = env.step(action)
    env.render()


