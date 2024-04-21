# The example followed here came from https://markelsanz14.medium.com/introduction-to-reinforcement-learning-part-3-q-learning-with-neural-networks-algorithm-dqn-1e22ee928ecd

from collections import deque
import gym
import tensorflow as tf
import numpy as np
import os

env = gym.make("CartPole-v1", render_mode="human")
num_features = env.observation_space.shape[0]
num_actions = env.action_space.n
env.action_space.seed(42)
observation, info = env.reset(seed=42)

# we are setting up a manual model in order to explicitly defien the feed forward/backward steps
class DQN(tf.keras.Model):
  """Dense neural network class."""
  def __init__(self):
    super(DQN, self).__init__()
    self.dense1 = tf.keras.layers.Dense(16, activation="relu")
    self.dense2 = tf.keras.layers.Dense(8, activation="relu")
    self.dense3 = tf.keras.layers.Dense(16, activation="relu")
    self.dense4 = tf.keras.layers.Dense(num_actions, dtype=tf.float32) # No activation
    
  def call(self, x):
    """Forward pass."""
    x = self.dense1(x)
    x = self.dense2(x)
    x = self.dense3(x)
    return self.dense4(x)

main_nn = DQN()
target_nn = DQN()
main_nn.load_weights('dqn_example_x.weights.h5')
target_nn.load_weights('dqn_example_x.weights.h5')

target_nn.save_weights(os.path.join(os.getcwd(), 'dqn_example_x.weights.h5'))
optimizer = tf.keras.optimizers.Adam(1e-4)
mse = tf.keras.losses.MeanSquaredError()
discount = 0.99

class ReplayBuffer(object):
    """ This is a buffer that will fill up with states + actions from which the network will learn from
        The buffer will grow in realtime.
    """
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, num_samples): # num samples here is the batch size which we will be training over
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idx = np.random.choice(len(self.buffer), num_samples)
        for i in idx: # we randomly select batches from the buffer which have states, actions, rewards from taking the actions, and the states appended 
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
buffer = ReplayBuffer(100000)

@tf.function
def train_step(states, actions, rewards, next_states, dones):
    """Perform a training iteration on a batch of data sampled from the experience
    replay buffer."""
    # Calculate targets.
    next_qs = target_nn(next_states) # find q values
    max_next_qs = tf.reduce_max(next_qs, axis=-1) # we find values which yield the best running reward, and we want to fit NN to it
    target = rewards + (1. - dones) * discount * max_next_qs
    with tf.GradientTape() as tape:
        qs = main_nn(states)
        action_masks = tf.one_hot(actions, num_actions) # converts to 32 x 2 
        masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
        loss = mse(target, tf.reduce_max(masked_qs, axis=-1))
    grads = tape.gradient(loss, main_nn.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))
    return loss

def select_epsilon_greedy_action(state, epsilon):
    """Take random action with probability epsilon, else take best action."""
    result = np.random.random()
    if result < epsilon:
        # Random action (left or right).
        return np.array(env.action_space.sample()) 
    else:
         # Greedy action for state. The model will return the action most likley to succeed
        return np.array(tf.argmax(main_nn(np.array(np.array([state])))[0]))


def main():
    num_episodes = 1000
    epsilon = 0.5
    batch_size = 128
    cur_frame = 0    
    last_100_ep_rewards = []
    for episode in range(num_episodes+1):     
        state = env.reset() # this will be reset whenever we are done, meaning we will just start over
        if isinstance(state, tuple):
           state = state[0]
        state = np.concatenate((state[:2], [0], state[2:], [0])) # adding horisontal accelertion and angular acceleration
        # state = np.array([0,0,0,0])
        ep_reward, done, iters = 0, False, 0
        while not done:
            iters +=1
            action = select_epsilon_greedy_action(state, epsilon)            
            next_state, reward, done, truncated, info = env.step(action) # every simulation step we are rewarded
            reward*=reward
            # Save to experience replay.
            next_state = np.concatenate((next_state[:2], [next_state[1]-state[1]], next_state[2:], [next_state[3]-state[4]]))
            buffer.add(state, action, ep_reward, next_state, done)
            state = next_state
            cur_frame += 1
            # Copy main_nn weights to target_nn.
            if cur_frame % 1000 == 0:
                target_nn.set_weights(main_nn.get_weights())        
        # Train neural network.
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            loss = train_step(states, actions, rewards, next_states, dones)
            print(episode, iters, cur_frame, loss)

        if episode < 950:
            epsilon -= 0.0005

        if len(last_100_ep_rewards) == 100:
            last_100_ep_rewards = last_100_ep_rewards[1:]
        last_100_ep_rewards.append(ep_reward)
            
        if episode % 50 == 0:
            print(f'Episode {episode}/{num_episodes}. Epsilon: {epsilon:.3f}. '
                f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.3f}')
            # saving weights once in awhile
            target_nn.save_weights(os.path.join(os.getcwd(), 'dqn_example_x.weights.h5'))
    env.close()


if __name__ == '__main__':
   main()