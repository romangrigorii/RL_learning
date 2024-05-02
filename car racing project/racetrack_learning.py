

# # # SETTING UP NN STUFF
num_actions = 8 # accelertion in the range -2,-1,0,1,2 and omega in the range -.02, -.01, 0, .01, .02 => this values map to action values 0-4 and 0-4
num_states = 8 # speed, acceleration, omega, and 5 distances in front of the car: -60 degrees, -30 degrees, 0 degrees, 30 degrees, 60 degrees
# # # 
from collections import deque
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from car_model import * 
from algorithmdevel import *
# we are setting up a manual model in order to explicitly defien the feed forward/backward steps
class DQN(tf.keras.Model):
  """Dense neural network class."""
  def __init__(self):
    super(DQN, self).__init__()
    self.dense1 = tf.keras.layers.Dense(128, activation="relu")
    self.dense2 = tf.keras.layers.Dense(128, activation="relu")
    self.dense3 = tf.keras.layers.Dense(num_actions, dtype=tf.float32) # No activation
  def call(self, x):
    """Forward pass."""
    x = self.dense1(x)
    x = self.dense2(x)
    return self.dense3(x)

class HyperParams():
  main_nn = DQN()
  target_nn = DQN()
  if 0: # set to 0 if you want to load fresh weights. WRANING, the saved weights will be overwritten
    main_nn.load_weights('dqn_racecar.weights.h5')
    target_nn.load_weights('dqn_racecar.weights.h5')
    # print("Saved weights loaded")
    epsilon = .1
    epsilon_step = 0
  else:
    print("Using random weights")
    epsilon = 1
    epsilon_step = .0009
  batch_size = 128

optimizer = tf.keras.optimizers.Adam(1e-4)
mse = tf.keras.losses.MeanSquaredError()

class ReplayBuffer(object):
  """Experience replay buffer that samples uniformly."""
  def __init__(self, size):
    self.buffer = deque(maxlen=size)

  def add(self, states, actions, rewards, next_states, dones):
    for i in range(len(states)):
      self.buffer.append((states[i], actions[i], rewards[i], next_states[i], dones[i]))

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
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards, dtype=np.float32)
    next_states = np.array(next_states)
    dones = np.array(dones, dtype=np.float32)
    return states, actions, rewards, next_states, dones

discount = 0.99

@tf.function
def train_step(states, actions, rewards, next_states, dones, hp):
  """Perform a training iteration on a batch of data sampled from the experience
  replay buffer."""
  # Calculate targets.
  next_qs = hp.target_nn(next_states)
  max_next_qs = tf.reduce_max(next_qs, axis=-1)
  target = rewards + (1. - dones) * discount * max_next_qs
  with tf.GradientTape() as tape:
    qs = hp.main_nn(states)
    action_masks = tf.one_hot(actions, num_actions)
    masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
    loss = mse(target, masked_qs)
  grads = tape.gradient(loss, hp.main_nn.trainable_variables)
  optimizer.apply_gradients(zip(grads, hp.main_nn.trainable_variables))
  return loss

def select_epsilon_greedy_action(state, hp):
  """Take random action with probability epsilon, else take best action."""
  result = tf.random.uniform((1,))
  if result < hp.epsilon:
    return [np.random.randint(0,5),np.random.randint(0,5)]
  else:
    sample = hp.main_nn(state)[0]
    return [tf.argmax(sample[:5]).numpy(),tf.argmax(sample[5:]).numpy()] # Greedy action for state.

def main():
  num_episodes = 1000    
  buffer = ReplayBuffer(100000)
  cur_frame = 0
  hp = HyperParams()
  rt = RaceTrack(10) # add 10 cars
  rt.reset_carrender()
  #print([[car.pos_x, car.pos_y] for car in rt.cars])
  # Start training. Play game once and then train with a batch.
  last_100_ep_rewards = []
  for episode in range(num_episodes+1):
    dones, ep_rewards, states = rt.extract_states()
    while sum(dones)==0: # train continuously

      plt.imshow(rt.carrender)
      plt.draw()
      plt.pause(0.1)
      plt.clf()

      state_in = [tf.expand_dims(state, axis=0) for state in states]
      actions = [select_epsilon_greedy_action(state, hp) for state in state_in]
      dones, rewards, next_states = rt.forward_sim(actions)
      ep_rewards = [rewards[i] + ep_rewards[i] +  next_states[i][0] for i in range(len(rt.cars))] # we are adding time spent without collision + velocity as reward
      # Save to experience replay.
      buffer.add(states, actions, ep_rewards, next_states, dones)
      states = next_states
      cur_frame += 1
      # Copy main_nn weights to target_nn.
      if cur_frame % 2000 == 0:
        hp.target_nn.set_weights(hp.main_nn.get_weights())

      # Train neural network.
      if len(buffer) >= hp.batch_size:
        states, actions, rewards, next_states, dones = buffer.sample(hp.batch_size)
        loss = train_step(states, actions, rewards, next_states, dones, hp)
    
    if episode < 950: 
      hp.epsilon -= hp.epsilon_step

    if len(last_100_ep_rewards) == 100:
      last_100_ep_rewards = last_100_ep_rewards[1:]
    last_100_ep_rewards.append(extract_list(ep_rewards, dones))
      
    if episode % 50 == 0 and episode>0:
      print(f'Episode {episode}/{num_episodes}. Epsilon: {hp.epsilon:.3f}. '
            f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.3f}')
      hp.main_nn.save_weights(os.path.join(os.getcwd(), 'dqn_example_original.weights.h5'))

if __name__ == '__main__':
  main()
