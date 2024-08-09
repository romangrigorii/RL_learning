import numpy as np

# # # SETTING UP NN STUFF
num_actions = (5, 5) # accelertion in the range -2,-1,0,1,2 and omega in the range -.02, -.01, 0, .01, .02 => this values map to action values 0-4 and 5-9
num_actions_tot = np.prod(num_actions)
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
from itertools import compress
# we are setting up a manual model in order to explicitly defien the feed forward/backward steps
class DQN(tf.keras.Model):
  """Dense neural network class."""
  def __init__(self):
    super(DQN, self).__init__()
    self.dense1 = tf.keras.layers.Dense(64, activation="relu")
    self.dense2 = tf.keras.layers.Dense(64, activation="relu")
    self.dense3 = tf.keras.layers.Dense(num_actions_tot, dtype=tf.float32) # No activation
  def call(self, x):
    """Forward pass."""
    x = self.dense1(x)
    x = self.dense2(x)
    return self.dense3(x)

class HyperParams():
  main_nn = DQN()
  target_nn = DQN()
  if 0: # set to 0 if you want to load fresh weights. WARNING, the saved weights will be overwritten
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

# @tf.function
# def train_step(states, actions, rewards, next_states, dones, hp):
#   """Perform a training iteration on a batch of data sampled from the experience
#   replay buffer."""
#   # Calculate targets.
#   next_qs = hp.target_nn(next_states)
#   max_next_qs = tf.concat([tf.expand_dims(tf.reduce_max(next_qs[:,:5], axis=-1), 1), tf.expand_dims(tf.reduce_max(next_qs[:,5:], axis=-1), 1)], axis = 1)  
#   rewards = tf.expand_dims(rewards, axis = 1)*tf.ones((1, 2))
#   dones = tf.expand_dims(dones, axis = 1)*tf.ones((1, 2))
#   target = rewards*tf.ones((len(rewards), 2))  + (1. - dones) * discount * max_next_qs
#   with tf.GradientTape() as tape:
#     qs = hp.main_nn(states)
#     action_masks_1 = tf.one_hot(actions[:,0], int(num_actions/2))
#     action_masks_2 = tf.one_hot(actions[:,1], int(num_actions/2))
#     masked_qs = tf.reduce_sum(tf.concat([action_masks_1, action_masks_2], axis = 1 )* qs, axis=-1)
#     loss = mse(tf.reduce_sum(target, axis=-1), masked_qs)
#     grads = tape.gradient(loss, hp.main_nn.trainable_variables)
#     optimizer.apply_gradients(zip(grads, hp.main_nn.trainable_variables))
#     return loss

# @tf.function
# def train_step(states, actions, rewards, next_states, dones, hp):
#   """Perform a training iteration on a batch of data sampled from the experience
#   replay buffer."""
#   # Calculate targets.
#   next_qs = hp.target_nn(next_states)
#   rewards = tf.expand_dims(rewards, axis = 1)*tf.ones((1, 10))
#   dones = tf.expand_dims(dones, axis = 1)*tf.ones((1, 10))
#   target = rewards*tf.ones((len(rewards), 10))  + (1. - dones) * discount * next_qs
#   with tf.GradientTape() as tape:
#     qs = hp.main_nn(states)
#     loss = mse(target, qs)
#     grads = tape.gradient(loss, hp.main_nn.trainable_variables)
#     optimizer.apply_gradients(zip(grads, hp.main_nn.trainable_variables))
#     return loss

@tf.function
def train_step(states, actions, rewards, next_states, dones, hp):
  """Perform a training iteration on a batch of data sampled from the experience
  replay buffer."""
  # Calculate targets.
  next_qs = hp.target_nn(next_states)                       # get the num_actions x batch_num matrix out corresponding to reward associated with each action
  max_next_qs = tf.reduce_max(next_qs, axis=-1)             # we pick out the max rewards
  target = rewards + (1. - dones) * discount * max_next_qs  # we take the current rewards and apply the reward equation to compute the new rewards
  with tf.GradientTape() as tape:                           
    qs = hp.main_nn(states)                                 # we take current state and forward propagate and pick most valuable actions
    action_masks = tf.one_hot(actions[:,0]*num_actions[0] + actions[:,1], num_actions_tot)         # one_hot converts from actions of the form [-1,1,1,-1...] to [[1,0],[0,1],[0,1]] essentally mapping actions with degree n to lists of length n
    masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)   # we multiply and sum across rows
    loss = mse(target, masked_qs)                           # we compute mse between two entities
  grads = tape.gradient(loss, hp.main_nn.trainable_variables) # now we find the gradient descent on the loss function with trainnable variables being modified
  optimizer.apply_gradients(zip(grads, hp.main_nn.trainable_variables)) # we apply the gradients
  return loss
  
def select_epsilon_greedy_action(state, hp):
  """Take random action with probability epsilon, else take best action."""
  result = tf.random.uniform((1,))
  if result < hp.epsilon:
    return [np.random.randint(0,num_actions[0]),np.random.randint(0,num_actions[0])]
  else:
    sample = hp.main_nn(state)[0]
    return [tf.argmax(sample).numpy()//num_actions[0],tf.argmax(sample).numpy() - tf.argmax(sample).numpy()//num_actions[0]] # Greedy action, acceleration and omega

def main():
  num_episodes = 100000    
  buffer = ReplayBuffer(10000)
  cur_frame = 0
  hp = HyperParams()
  rt = RaceTrack(10) # add 10 cars
  rt.reset_carrender()
  #print([[car.pos_x, car.pos_y] for car in rt.cars])
  # Start training. Play game once and then train with a batch.
  last_100_ep_rewards = deque(maxlen=100)
  episode = 0
  dones, ep_rewards, states = rt.extract_states()
  while episode < num_episodes+1:   
    # dones, ep_rewards, states = rt.extract_states()
    episode += sum(dones) # an episode is over when acar collides

    plt.imshow(rt.carrender)
    plt.draw()
    plt.pause(0.1)
    plt.clf()

    state_in = [tf.expand_dims(state, axis=0) for state in states]
    if episode < 100:
      trn = [min(max(int((s[7]-s[3] + np.random.randn()*20)/20),-2), 2)+2 for s in states]
      action = [[4 if states[q][0]<100 else 2, trn[q]] for q in range(len(states))]
    else:
      actions = [select_epsilon_greedy_action(state, hp) for state in state_in]
    dones, rewards, next_states = rt.forward_sim(action)
    ep_rewards = [rewards[i] + ep_rewards[i] for i in range(len(rt.cars))] # we are adding time spent without collision + velocity as reward
    # Save to experience replay.
    
    buffer.add(next_states, action, ep_rewards, next_states, dones)
    cur_frame += 1
    # Copy main_nn weights to target_nn.
    if cur_frame % 500 == 0:
      hp.target_nn.set_weights(hp.main_nn.get_weights())      
    # Train neural network.
    if len(buffer) >= hp.batch_size:
      states, actions, rewards, next_states, dones_ = buffer.sample(hp.batch_size)
      # print(states.shape, actions.shape, rewards.shape)
      loss = train_step(states, actions, rewards, next_states, dones_, hp)

    hp.epsilon = max(.1, 1 - episode/1000)

    last_100_ep_rewards.extend(list(compress(ep_rewards, dones)))
    #print(last_100_ep_rewards)
    # print(episode)
    if episode % 20 == 0 and episode>0:
      print(f'Episode {episode}/{num_episodes}. Epsilon: {hp.epsilon:.3f}. '
            f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.3f}')
      hp.main_nn.save_weights(os.path.join(os.getcwd(), 'dqn_example_original.weights.h5'))

if __name__ == '__main__':
  main()
