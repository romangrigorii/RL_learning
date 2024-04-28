# The example followed here came from https://markelsanz14.medium.com/introduction-to-reinforcement-learning-part-3-q-learning-with-neural-networks-algorithm-dqn-1e22ee928ecd

import gym
from collections import deque
import tensorflow as tf
import numpy as np
import random
import math
import time
import glob
import io
import os

env = gym.make('CartPole-v0', render_mode="human")
num_features = env.observation_space.shape[0] + 2
num_actions = env.action_space.n
print('Number of state features: {}'.format(num_features))
print('Number of possible actions: {}'.format(num_actions))

# we are setting up a manual model in order to explicitly defien the feed forward/backward steps
class DQN(tf.keras.Model):
  """Dense neural network class."""
  def __init__(self):
    super(DQN, self).__init__()
    self.dense1 = tf.keras.layers.Dense(32, activation="relu")
    self.dense2 = tf.keras.layers.Dense(32, activation="relu")
    self.dense3 = tf.keras.layers.Dense(num_actions, dtype=tf.float32) # No activation
    
  def call(self, x):
    """Forward pass."""
    x = self.dense1(x)
    x = self.dense2(x)
    return self.dense3(x)

main_nn = DQN()
target_nn = DQN()
global epsilon, epsilon_step

class HyperParams():
  if 0: # set to 0 if you want to load fresh weights. WRANING, the saved weights will be overwritten
    main_nn.load_weights('cart_balanncing_enhanced.weights.h5')
    target_nn.load_weights('cart_balanncing_enhanced.weights.h5')
    print("Saved weights loaded")
    epsilon = .1
    epsilon_step = 0
  else:
    print("Using random weights")
    epsilon = 1
    epsilon_step = .001
  batch_size = 128

optimizer = tf.keras.optimizers.Adam(1e-4)
mse = tf.keras.losses.MeanSquaredError()

class ReplayBuffer(object):
  """Experience replay buffer that samples uniformly."""
  def __init__(self, size):
    self.buffer = deque(maxlen=size)

  def add(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def __len__(self):
    return len(self.buffer)

  def sample(self, num_samples):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    idx = np.random.choice(len(self.buffer), num_samples)
    for i in idx:
      elem = self.buffer[i]
      state, action, reward, next_state, done = elem
      states.append(np.array(state, copy=False))
      actions.append(np.array(action, copy=False))
      rewards.append(reward)
      next_states.append(np.array(next_state, copy=False))
      dones.append(done)
    states = np.array(states, dtype="object")
    actions = np.array(actions)
    rewards = np.array(rewards, dtype=np.float32)
    next_states = np.array(next_states)
    dones = np.array(dones, dtype=np.float32)
    return states, actions, rewards, next_states, dones

discount = 0.99

@tf.function
def train_step(states, actions, rewards, next_states, dones):
  """Perform a training iteration on a batch of data sampled from the experience
  replay buffer."""
  # Calculate targets.
  next_qs = target_nn(next_states)
  max_next_qs = tf.reduce_max(next_qs, axis=-1)
  target = rewards + (1. - dones) * discount * max_next_qs
  with tf.GradientTape() as tape:
    qs = main_nn(states)
    action_masks = tf.one_hot(actions, num_actions)
    masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
    loss = mse(target, masked_qs)
  grads = tape.gradient(loss, main_nn.trainable_variables)
  optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))
  return loss

def select_epsilon_greedy_action(state, epsilon):
  """Take random action with probability epsilon, else take best action."""
  result = tf.random.uniform((1,))
  if result < epsilon:
    return env.action_space.sample() # Random action (left or right).
  else:
    return tf.argmax(main_nn(state)[0]).numpy() # Greedy action for state.

def main():
  num_episodes = 1000    
  buffer = ReplayBuffer(100000)
  cur_frame = 0
  hp = HyperParams()
  # Start training. Play game once and then train with a batch.
  last_100_ep_rewards = []
  for episode in range(num_episodes+1):
    state = np.array(env.reset()[0])
    ep_reward, done = 0, False
    while not done:
      state_in = tf.expand_dims(state, axis=0)
      action = select_epsilon_greedy_action(state_in, hp.epsilon)
      q = env.step(action)
      next_state, reward, done, _ , _  = env.step(action)
      ep_reward += reward
      next_state = np.concatenate((next_state[:2], [next_state[1]-state[1]], next_state[2:], [next_state[3]-state[3]])) # adding acceleration 
      print(next_state)
      # Save to experience replay.
      buffer.add(state, action, reward, next_state, done)
      state = next_state
      cur_frame += 1
      # Copy main_nn weights to target_nn.
      if cur_frame % 2000 == 0:
        target_nn.set_weights(main_nn.get_weights())

      # Train neural network.
      if len(buffer) >= hp.batch_size:
        states, actions, rewards, next_states, dones = buffer.sample(hp.batch_size)
        loss = train_step(states, actions, rewards, next_states, dones)
    
    if episode < 950:
      hp.epsilon -= hp.epsilon_step

    if len(last_100_ep_rewards) == 100:
      last_100_ep_rewards = last_100_ep_rewards[1:]
    last_100_ep_rewards.append(ep_reward)
      
    if episode % 50 == 0:
      print(f'Episode {episode}/{num_episodes}. Epsilon: {hp.epsilon:.3f}. '
            f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.3f}')
      target_nn.save_weights(os.path.join(os.getcwd(), 'dqn_example_w_accelerations.weights.h5'))
  env.close()


if __name__ == '__main__':
  main()