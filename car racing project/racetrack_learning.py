# # in this module we will attempt to get a car to follow a racetrack

# # available actions = turn left, turn_right, accelerate, decelarate.
# # simultion will work at 20ms intervals, 20ms = 1 sim
# # We can at maximum turn .02 radians in 1 sim -> 1 radian/s -> .02/sim
# # Speed is locked between 0 and 500 pixels/s -> 10pixels/sim which is roughtly 110 mph
# # we can add/subtract at maximum 2 pixels from our velocity in a simulation loop, which corresponds to 
# # 10 pixels = 1m
# # avaialble states = distance to the wall in front of the car at 60 degrees, 30 degrees, 0 degrees, -30 degrees, -60 degrees
# # on the racerack, cars will be mapped with -1, the allowed track will be marked as 0, and walls of the track wll be marked with a 1

import scipy 
from collections import deque
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

# # # SETTING UP CAR STUFF

if 1:
  image = img.imread('car racing racetrack2/racetrack.png')
  raceTrack = np.matrix([[1 if q[0] <=.2 else 0 for q in c] for c in image])
else:
  image = img.imread('car racing racetrack1/racetrack.png')
  raceTrack = np.matrix([[2 if q[0]>.5 and q[1] < .5 and q[2]< .5 else 1 if q[0] == 0 and q[1] == 0 and q[2] == 0 else 0 for q in c] for c in image])

(Ymax, Xmax) = raceTrack.shape
print(Ymax, Xmax)

class Car:
  # note that car kinematics are tied to the center of the car
  def __init__(self):
    self.car_width = 15 # 1.5m 
    self.car_length = 27 # 2.7m
    self.pos_x = 92
    self.pos_y = 195
    self.speed = 0
    self.acc = 0
    self.theta = np.pi/2 # rotation
    self.omega = 0 # rotational velocity
    self.collided = False

    # static
    self.dt = 0.5
    self.top_speed = 500 # pixels per second
    self.top_acc = 20 # pixels per second per second. can be +-
    self.top_omega = 1 # radian per second

    def sample_randomly(self):
      return (np.random.rand - .5)*self.top_acc*2, (np.random.rand - .5)*self.omega*2

class RaceTrack(Car):
  def __init__(self, num_cars = 1):
    self.racetrack = raceTrack
    self.carrender = np.zeros(np.shape(raceTrack))
    self.num_cars = num_cars
    self.cars = [Car() for q in range(num_cars)]
    
  def reset_carrender(self):
    self.carrender = self.racetrack.copy()

  def reset_car(self, idx): # will reset a given car
    try: 
      assert idx<self.num_cars
    except:
      Exception("you tried to access a car that doesn't exist")
    self.cars[idx].__init__()
  
  def position_car(self, idx):
    car = self.cars[idx]
    for x in range(-car.car_width//2, car.car_width//2+1):
      for y in range(-car.car_length//2, car.car_length//2+1):
        x_ = car.pos_x - x*np.sin(car.theta) + y*np.cos(car.theta)
        y_ = car.pos_y + x*np.cos(car.theta) + y*np.sin(car.theta)
        x_, y_ = int(x_), int(y_)
        if x_>=0 and x_<Xmax and y_>=0 and y_<Ymax:
          if self.racetrack[y_,x_] == 1: 
            car.collided = True
          self.carrender[y_,x_] = -idx - 1
    return car.collided


  def position_cars(self):
    for i, b in enumerate([self.position_car(i) for i in range(len(self.cars))]):
      if b: self.reset_car(i)


  def forward_sim(self, state): # delta state is new acceleration and new angular velocity
    for car in self.cars:
      car.omega = state[1]
      if car.omega>car.top_omega: car.omega = car.top_omega
      if car.omega<-car.top_omega: car.omega = -car.top_omega
      car.theta += car.dt * car.omega
      car.acc = state[0]
      if car.acc > car.top_acc: car.acc = car.top_acc
      if car.acc < -car.top_acc: car.acc = -car.top_acc
      car.speed += car.acc*car.dt
      car.pos_x += car.speed*car.dt*np.cos(car.theta)
      car.pos_y += car.speed*car.dt*np.sin(car.theta)

# # # SETTING UP NN STUFF
num_actions = 2 # we either accelerate (assuming instantenous acceleration) or turn the car (assuming steering wheel can be set instanteneously)
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

def select_epsilon_greedy_action(state, hp, car):
  """Take random action with probability epsilon, else take best action."""
  result = tf.random.uniform((1,))
  if result < hp.epsilon:
    return car.sample_randomly() # Random action (left or right).
  else:
    return tf.argmax(hp.main_nn(state)[0]).numpy() # Greedy action for state.

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
      action = select_epsilon_greedy_action(state_in, hp)
      q = env.step(action)
      next_state, reward, done, _ , _  = env.step(action)
      ep_reward += reward
      # Save to experience replay.
      buffer.add(state, action, reward, next_state, done)
      state = next_state
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
    last_100_ep_rewards.append(ep_reward)
      
    if episode % 50 == 0 and episode>0:
      print(f'Episode {episode}/{num_episodes}. Epsilon: {hp.epsilon:.3f}. '
            f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.3f}')
      hp.main_nn.save_weights(os.path.join(os.getcwd(), 'dqn_example_original.weights.h5'))
  env.close()


if __name__ == '__main__':
  main()
if __name__ == "__main__":
  rt = RaceTrack(10)
  for i in range(10):
    rt.cars[i].theta = np.random.rand(1)*2*np.pi 
  while 1:
    rt.reset_carrender()
    rt.position_cars()
    # print(rt.cars[0].speed)
    # print(rt.cars[0].acc)
    plt.imshow(rt.carrender)
    plt.draw()
    plt.pause(0.1)
    plt.clf()
    rt.forward_sim([1,0])