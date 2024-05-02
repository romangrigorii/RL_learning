# The example followed here came from https://markelsanz14.medium.com/introduction-to-reinforcement-learning-part-3-q-learning-with-neural-networks-algorithm-dqn-1e22ee928ecd
# 
# 
# The state space of the cart is: [x pos of vart, x velocity of cart, angle of bar, angular vel of bar]
import gym
from collections import deque
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
import bisect

env = gym.make('CartPole-v0', render_mode="human")
num_features = env.observation_space.shape[0]
num_actions = env.action_space.n
print('Number of state features: {}'.format(num_features))
print('Number of possible actions: {}'.format(num_actions))

# we are setting up a manual model in order to explicitly defien the feed forward/backward steps
@tf.keras.utils.register_keras_serializable('my_package')
class DQN(tf.keras.Model):
  """Dense neural network class."""
  def __init__(self, **kwargs):
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
  use_saved_model = True # note that new data will be saved only when we don't use a saved model
  save_weights = True # not use_saved_model
  model_name = 'cart_balancing_model_128_128_q.keras'  
  num_iters = 10000
  epslion_convergence_steps = 1000
  epsilon_min = .1 # the minimum epsilon that system will use after epslion_convergence_steps sim runs
  if use_saved_model: # set to 0 if you want to load fresh weights. WRANING, the saved weights will be overwritten
    main_nn = tf.keras.models.load_model(model_name)
    target_nn = tf.keras.models.load_model(model_name)
    # print("Saved weights loaded")
    epsilon = .1
    epsilon_step = 0
  else:
    main_nn = DQN()
    target_nn = DQN()
    print("Using random weights")
    epsilon = 1.0
    epsilon_step = (epsilon-epsilon_min)/epslion_convergence_steps
  batch_size = 128

optimizer = tf.keras.optimizers.Adam(1e-4)
mse = tf.keras.losses.MeanSquaredError()

class ReplayBuffer(object):
  """Experience replay buffer that samples uniformly."""
  def __init__(self, size):
    self.buffer = deque(maxlen=size)

  def add(self, state, action, reward, next_state, done):
    # q = bisect.bisect_right(self.buffer,reward, key=lambda x: x[2])
    # self.buffer.insert(q, (state, action, reward, next_state, done))
    self.buffer.append((state, action, reward, next_state, done)) # add a tuple of data to the buffer 

  def __len__(self):
    return len(self.buffer)

  def sample(self, num_samples):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    idx = np.random.choice(len(self.buffer), num_samples)
    #idx = [len(self.buffer) - i for i in range(1, num_samples+1)]
    for i in idx:
      elem = self.buffer[i]
      state, action, reward, next_state, done = elem
      states.append(np.array(state, copy=False))
      actions.append(np.array(action, copy=False))
      rewards.append(reward)
      next_states.append(np.array(next_state, copy=False))
      dones.append(done)      
    # plt.plot(rewards)
    # plt.show()
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
  next_qs = hp.target_nn(next_states)                       # get the 2 x batch_num matrix out corresponding to reward associated with each action
  max_next_qs = tf.reduce_max(next_qs, axis=-1)             # we pick out the max rewards
  target = rewards + (1. - dones) * discount * max_next_qs  # we take the current rewards and apply the reward equation to compute the new rewards
  with tf.GradientTape() as tape:                           
    qs = hp.main_nn(states)                                 # we take current state and forward propagate and pick most valuable actions
    action_masks = tf.one_hot(actions, num_actions)         # one_hot converts from actions of the form [-1,1,1,-1...] to [[1,0],[0,1],[0,1]] essentally mapping actions with degree n to lists of length n
    masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)   # we multiply and sum across rows
    loss = mse(target, masked_qs)                           # we compute mse between two entities
    grads = tape.gradient(loss, hp.main_nn.trainable_variables) # now we find the gradient descent on the loss function with trainnable variables being modified
    optimizer.apply_gradients(zip(grads, hp.main_nn.trainable_variables)) # we apply the gradients
    return loss

def select_epsilon_greedy_action(state, hp):
  """Take random action with probability epsilon, else take best action."""
  result = tf.random.uniform((1,))
  if result < hp.epsilon:
    return env.action_space.sample() # Random action (left or right).
  else:
    return tf.argmax(hp.main_nn(state)[0]).numpy() # Greedy action for state.

def main():  
  buffer = ReplayBuffer(100000)
  cur_frame = 0
  hp = HyperParams()
  num_episodes = hp.num_iters
  # Start training. Play game once and then train with a batch.
  last_100_ep_rewards = []
  for episode in range(num_episodes+1):
    state = np.array(env.reset()[0])                          # this is a default state
    ep_reward, done = 0, False                                
    while not done:
      state_in = tf.expand_dims(state, axis=0)                # I add a dimension here to make it a 1 x 4 from ,4
      action = select_epsilon_greedy_action(state_in, hp)     # we select an action based on epsilon criteria
      next_state, reward, done, _ , _  = env.step(action)
      ep_reward += (reward - abs(next_state[0])/5.0)          # we reward based on time spent with upright stick and how close it was to the middle
      # Save to experience replay.
      buffer.add(state, action, ep_reward, next_state, done)  # we save the state and the chose action reward and the next state we find ourselves in
      state = next_state
      cur_frame += 1
      # Copy main_nn weights to target_nn.
      if cur_frame % 10000 == 0: # once we collect enough new data we
        hp.target_nn.set_weights(hp.main_nn.get_weights())

      # Train neural network.
      if len(buffer) >= hp.batch_size:
        states, actions, rewards, next_states, dones = buffer.sample(hp.batch_size)
        loss = train_step(states, actions, rewards, next_states, dones, hp)
    
    if hp.epsilon>hp.epsilon_min:
      hp.epsilon -= hp.epsilon_step

    if len(last_100_ep_rewards) == 100:
      last_100_ep_rewards = last_100_ep_rewards[1:]
    last_100_ep_rewards.append(ep_reward)
      
    if episode % 50 == 0 and episode>0:
      print(f'Episode {episode}/{num_episodes}. Epsilon: {hp.epsilon:.3f}. '
            f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.3f}')
      if hp.save_weights:
        hp.main_nn.save(os.path.join(os.getcwd(), hp.model_name))
  env.close()


if __name__ == '__main__':
  main()