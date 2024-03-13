# # in this module we will attempt to get a car to follow a racetrack

# # available actions = turn left, turn_right, accelerate, decelarate, (speeds are 0 through 10)
# # avaialble states = speed_x, speed_y, pos_x, pos_y, angle_x, angle_y

import scipy 
from collections import deque
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.image as img
image = img.imread('racetrack.png')
print(image.shape)
raceTrack = [[2 if q[0]>.5 and q[1] < .5 and q[2]< .5 else 1 if q[0] == 0 and q[1] == 0 and q[2] == 0 else 0 for q in c] for c in image]

actions = [0,1,2,3]
num_actions = len(actions)

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
  

class ReplayBuffer(object):
  """Experience replay buffer that samples uniformly."""
  def __init__(self, size):
    self.buffer = deque(maxlen=size)

  def add(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def __len__(self):
    return len(self.buffer)

  def sample(self, num_samples): # num samples here is the batch size which we will be training over
    states, actions, rewards, next_states, dones = [], [], [], [], []
    idx = np.random.choice(len(self.buffer), num_samples)
    for i in idx: # we randomly select random batches from the buffer which have states, actions, rewards from taking the actions, and the states appended 
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
  
main_nn = DQN()
target_nn = DQN()
optimizer = tf.keras.optimizers.Adam(1e-4)
mse = tf.keras.losses.MeanSquaredError()
discount = 0.99

@tf.function
def train_step(states, actions, rewards, next_states, dones):
  """Perform a training iteration on a batch of data sampled from the experience
  replay buffer."""
  # Calculate targets.
  next_qs = target_nn(next_states) # find q values 
  max_next_qs = tf.reduce_max(next_qs, axis=-1)
  target = rewards + (1. - dones) * discount * max_next_qs
  print(target)
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
  result = np.random.random()
  acts = [4]*num_actions
  if result < epsilon:    
    acts[np.random.randint(num_actions)] = 1
    return acts # Random action (left or right).
  else:
    acts[np.int(tf.argmax(main_nn(state))[0])] = 1
    return acts # Greedy action for state. The model will return the action most likley to succeed
  
def car_model(state, action):
    # # available actions = turn left, turn_right, accelerate, decelarate, (speeds are 0 through 10)
    # # avaialble states = pos_x, pos_y, angle, speed
    state_old = state.copy()
    if action[0] == 1:
        state[3] += .1
    if action[1] == 1:
        state[3] -= .1
    state[3] += action[2] - action[3]
    if state[2] > 10: state[2] = 10
    if state[2] < 0: state[2] = 0
    speed_x = np.sin(state[3])*state[2] + np.cos(state[3])*state[2]
    speed_y = np.cos(state[3])*state[2] - np.sin(state[3])*state[2]
    state[0] += speed_x
    state[1] += speed_y
    state[0] = int(state[0])
    state[1] = int(state[1])    
    if state[0]< 0 : state[0] = 0
    if state[0]>195: state[0] = 195
    if state[1]< 0 : state[1] = 0
    if state[1]>195: state[1] = 195
    reward = raceTrack[state[1]][state[0]]
    done = state[0]<45 and state[1]<93 and state[1]>73 and state_old[0]>45
    return state, reward, done

   

def main():
    num_episodes = 1000
    epsilon = 1.0
    batch_size = 32
    buffer = ReplayBuffer(100000)
    cur_frame = 0
    
    last_100_ep_rewards = []
    for episode in range(num_episodes+1):     
        state = [0, 0, 20, 55] # this will be reset whenever we are done, meaning we will just start over
        ep_reward, done, iters = 0, False, 0       

        while not done:
            iters +=1
            #print(state)
            action = select_epsilon_greedy_action(state, epsilon)            
            #print(env.step(action))
            next_state, reward, done = car_model(state, action)
            ep_reward += reward
            # Save to experience replay.
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            cur_frame += 1
            # Copy main_nn weights to target_nn.
            if cur_frame % 100 == 0:
                target_nn.set_weights(main_nn.get_weights())

            plt.imshow(raceTrack + [[2 if state[0]==q and state[1]==h else 0 for q in range(196)] for h in range(196)])

            # Train neural network.
            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                loss = train_step(states, actions, rewards, next_states, dones)

        print(episode, iters)

    if episode < 950:
        epsilon -= 0.001

    if len(last_100_ep_rewards) == 100:
        last_100_ep_rewards = last_100_ep_rewards[1:]
    last_100_ep_rewards.append(ep_reward)
        
    if episode % 50 == 0:
        print(f'Episode {episode}/{num_episodes}. Epsilon: {epsilon:.3f}. '
            f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.3f}')



if __name__ == '__main__':
   main()
