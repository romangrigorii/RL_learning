# # In this module we model a car and a racetrack
# # available actions = turn left, turn_right, accelerate, decelarate.
# # simultion will work at 20ms intervals, 20ms = 1 sim step
# # We can, at most turn .02 radians in 1 sim -> 1 radian/s -> .02/sim
# # 10 pixels = 1m
# # Speed is locked between 0 and 500 pixels/s -> 10 pixels/sim which is roughtly 110 mph
# # we can add/subtract at maximum 2 pixels from our velocity in a simulation loop, which corresponds to 100 pixels/s change
# # avaialble states = distance to the wall in front of the car at 60 degrees, 30 degrees, 0 degrees, -30 degrees, -60 degrees
# # avalable actions : acceleration -> -2 -1 0 1 2  omega: -.02 .01 0 .01 .02  (10 actions total)
# # on the racerack, cars will be mapped with -1, the race track will be marked as 0, and walls of the track wll be marked with a 1

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
# # # SETTING UP CAR STUFF
if 1:
  image = img.imread('racetrack2/racetrack.png')
  raceTrack = np.matrix([[1 if q[0] <=.2 else 0 for q in c] for c in image])
else:
  image = img.imread('racetrack1/racetrack.png')
  raceTrack = np.matrix([[2 if q[0]>.5 and q[1] < .5 and q[2]< .5 else 1 if q[0] == 0 and q[1] == 0 and q[2] == 0 else 0 for q in c] for c in image])

(Ymax, Xmax) = raceTrack.shape
print("The shape of ractrack is: ", Ymax, Xmax)

d_theta = [-np.pi/2, -np.pi/4, 0.0, np.pi/4, np.pi/2]
class Car:
  # note that car kinematics are tied to the center of the car
  def __init__(self):
    self.car_width = 15 # 1.5m 
    self.car_length = 27 # 2.7m
    self.pos_x = 90
    self.pos_y = 195
    self.speed = 0
    self.acc = 0
    self.theta = np.pi/2 # rotation
    self.omega = 0 # rotational velocity
    self.collided = False
    self.distances = [0]*5
    # static
    self.car_angle = 0
    self.car_angle_old = 0
    self.car_angle_diff = 0

    self.dt = 0.1
    self.top_speed = 500 # pixels per second
    self.top_acc = 20 # pixels per second per second. can be +-
    self.top_omega = 1 # radian per second

    self.possible_actions = [-100, -50, 0, 50, 100, -5, -2.5, 0.0, 2.5, 5] # these are scaled for dt = .02 - needs rescaling for different dt
    self.possible_actions = [q*self.dt for q in self.possible_actions[:5]] + self.possible_actions[5:]

    def sample_randomly(self):
      return (np.random.rand - .5)*self.top_acc*2, (np.random.rand - .5)*self.omega*2

class RaceTrack(Car):
  def __init__(self, num_cars = 1):
    self.racetrack = raceTrack
    self.racetrack_size = np.shape(raceTrack)
    self.carrender = np.zeros(self.racetrack_size)
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
    '''
    This will draw a car on the racetrack template. I don't think it's the best version - can be improved.
    It wil lalso return colided flag as True if there is a collision detected
    '''
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

  def compute_distances(self):
    '''
    This will compute the 5 distances between the car and the reacetrack wall
    '''
    distances = [[0]*5 for q in range(len(self.cars))]    
    for i, car in enumerate(self.cars):
        x, y = car.pos_x, car.pos_y
        for j, d_th in enumerate(d_theta):
            d_est = 0
            x_, y_ = int(x), int(y)
            while self.carrender[y_, x_] != 1:
                d_est += 2
                x_ = int(x + (d_est)*np.cos(car.theta + d_th))
                y_ = int(y + (d_est)*np.sin(car.theta + d_th))              
            distances[i][j] = d_est
    return distances

  def extract_states(self):
    dones = [0]*len(self.cars)
    reward = [q.car_angle_diff for q in self.cars]
    # print(reward)
    for i, b in enumerate([self.position_car(i) for i in range(len(self.cars))]):
      if b: 
        self.reset_car(i)
        dones[i] = 1
    distances = self.compute_distances()
    states = [[self.cars[i].speed, self.cars[i].acc, self.cars[i].omega] + distances[i] for i in range(len(self.cars))]
    return dones, reward, states


  def forward_sim_states(self, actions): # delta state is new acceleration and new angular velocity
    for i, car in enumerate(self.cars):
      car.omega = car.possible_actions[5 + actions[i][1]]
      if car.omega>car.top_omega: car.omega = car.top_omega
      if car.omega<-car.top_omega: car.omega = -car.top_omega
      car.theta += car.dt * car.omega
      car.acc = car.possible_actions[actions[i][0]]
      if car.acc > car.top_acc: car.acc = car.top_acc
      if car.acc < -car.top_acc: car.acc = -car.top_acc
      car.speed += car.acc*car.dt
      car.pos_x += car.speed*car.dt*np.cos(car.theta)
      car.pos_y += car.speed*car.dt*np.sin(car.theta)
      #print(car.pos_y)
      car.car_angle_old = car.car_angle
      car.car_angle = np.arctan2(self.racetrack_size[0]/2-car.pos_y, car.pos_x - self.racetrack_size[1]/2)
      car.car_angle_diff = car.car_angle - car.car_angle_old
      # car.car_angle_diff = car.car_angle_diff*(car.car_angle_diff>0)

  def forward_sim(self, actions):
      self.reset_carrender()
      self.forward_sim_states(actions)
      return self.extract_states()

if __name__ == "__main__":
  rt = RaceTrack(1) # add 10 cars
  states = [[0]*10 for i in range(len(rt.cars))]
  while 1:
    trn = [min(max(int((s[7]-s[3])/20),-2), 2)+2 for s in states]
    print(states)
    print([q.car_angle for q in rt.cars])
    states = rt.forward_sim([[4 if states[q][0]<100 else 2, trn[q]] for q in range(len(rt.cars))])[2] # the car is accelerating
    #print(states, '\n')
    plt.imshow(rt.carrender)
    plt.draw()
    plt.pause(0.1)
    plt.clf()
    # we create a suite of 10 cars with positive acceleration and randomized omega
    