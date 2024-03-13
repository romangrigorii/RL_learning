# The example followed here came from https://markelsanz14.medium.com/introduction-to-reinforcement-learning-part-3-q-learning-with-neural-networks-algorithm-dqn-1e22ee928ecd

from collections import deque
import gym
import tensorflow as tf
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")
num_features = env.observation_space.shape[0]
num_actions = env.action_space.n
env.action_space.seed(42)
observation, info = env.reset(seed=42)

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras.layers import Dropout

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

main_nn = Sequential([
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
  Dense(num_actions, dtype=tf.float32)
])

target_nn = Sequential([
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
  Dense(num_actions, dtype=tf.float32)
])

main_nn.compile(
  'adam',
  loss='MeanSquaredError',
  metrics=['accuracy'],
)

target_nn.compile(
  'adam',
  loss='MeanSquaredError',
  metrics=['accuracy'],
)

def fit_to_nn(train_i, train_o, test_i, test_o):
  print("hello")
  main_nn.fit(
    train_i,
    train_o,
    epochs=1,
    validation_data=(test_i,test_o))

class ReplayBuffer(object):
  """Experience replay buffer that samples uniformly."""
  def __init__(self, size):
    self.buffer = deque(maxlen=size)

  def add(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def __len__(self):
    return len(self.buffer)

  def sample(self, train_samples, test_samples): # num samples here is the batch size which we will be training over
    states, actions, rewards, next_states, dones = [], [], [], [], []
    states_v, actions_v, rewards_v, next_states_v, dones_v = [], [], [], [], []
    idx = np.random.choice(len(self.buffer), train_samples + test_samples)
    for i in idx[:train_samples]: # we randomly select random batches from the buffer which have states, actions, rewards from taking the actions, and the states appended 
      elem = self.buffer[i]
      state, action, reward, next_state, done = elem
      states.append(np.array(state))
      actions.append(np.array(action))
      rewards.append(reward)
      next_states.append(np.array(next_state))
      dones.append(done)
    for i in idx[train_samples:]: # we randomly select random batches from the buffer which have states, actions, rewards from taking the actions, and the states appended 
      elem = self.buffer[i]
      state, action, reward, next_state, done = elem
      states_v.append(state)
      actions_v.append(action)
      rewards_v.append(reward)
      next_states_v.append(next_state)
      dones_v.append(done)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    dones = np.array(dones)
    states_v = np.array(states_v)
    actions_v = np.array(actions_v)
    rewards_v = np.array(rewards_v)
    next_states_v = np.array(next_states_v)
    dones_v = np.array(dones_v)
    return (states, actions, rewards, next_states, dones), (states_v, actions_v, rewards_v, next_states_v, dones_v)
    #return states, actions, rewards, next_states, dones
  
discount = 0.99

@tf.function
def train_step(train, test):
  """Perform a training iteration on a batch of data sampled from the experience
  replay buffer."""
  # Calculate targets.
  states, actions, rewards, next_states, dones = train
  states_v, actions_v, rewards_v, next_states_v, dones_v = test

  next_qs = target_nn.predict(next_states) # find q values
  max_next_qs = np.max(next_qs, axis=-1)
  target = rewards + (1. - dones) * discount * max_next_qs
  # print(target)
  # qs = main_nn.predict(states)
  # action_masks = tf.one_hot(actions, num_actions)
  # masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)

  next_qs_v = target_nn(next_states_v) # find q values
  max_next_qs_v = np.max(next_qs_v, axis=-1)
  target_v = rewards_v + (1. - dones_v) * discount * max_next_qs_v
  # qs_v = main_nn.predict(states_v)
  # action_masks_v = tf.one_hot(actions_v, num_actions)
  # masked_qs_v = tf.reduce_sum(action_masks_v * qs_v, axis=-1)
  states = np.array(states)
  target = np.array(target)
  states_v = np.array(states_v)
  target_v = np.array(target_v)
  print(states)
  print(target)
  print(states_v)
  print(target_v)
  return fit_to_nn(states, target, states_v, target_v)

def select_epsilon_greedy_action(state, epsilon):
  """Take random action with probability epsilon, else take best action."""
  result = np.random.random()
  if result < epsilon:
    return np.array(env.action_space.sample()) # Random action (left or right).
  else:
    return np.array(tf.argmax(main_nn(np.array(np.array([state])))[0]))  # Greedy action for state. The model will return the action most likley to succeed
  

def main():
    num_episodes = 1000
    epsilon = 1.0
    batch_size = 320
    buffer = ReplayBuffer(100000)
    cur_frame = 0
    
    last_100_ep_rewards = []
    for episode in range(num_episodes+1):     
        state = env.reset() # this will be reset whenever we are done, meaning we will just start over
        if isinstance(state, tuple):
           state = state[0]
        # state = np.array([0,0,0,0])
        ep_reward, done, iters = 0, False, 0
        while not done:
            iters +=1
            #print(state)
            action = select_epsilon_greedy_action(state, epsilon)            
            #print(env.step(action))
            next_state, reward, done, truncated, info = env.step(action) # every simulation step we are rewarded
            ep_reward += reward
            # Save to experience replay.
            buffer.add(state, action, ep_reward, next_state, done)
            state = next_state
            cur_frame += 1
            # Copy main_nn weights to target_nn.
            if cur_frame % 1000 == 0:
                target_nn.set_weights(main_nn.get_weights())

            # Train neural network.
            if len(buffer) >= batch_size:
                train, test = buffer.sample(batch_size, 80)
                train_step(train, test)
        print(episode, iters, cur_frame)

    if episode < 950:
        epsilon -= 0.001

    if len(last_100_ep_rewards) == 100:
        last_100_ep_rewards = last_100_ep_rewards[1:]
    last_100_ep_rewards.append(ep_reward)
        
    if episode % 50 == 0:
        print(f'Episode {episode}/{num_episodes}. Epsilon: {epsilon:.3f}. '
            f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.3f}')
    env.close()


if __name__ == '__main__':
   main()