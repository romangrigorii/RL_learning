# # This will simulat a 1D environment where you are allowed to go forward or backward
# # the rewards are at the very end of the path, the goal is to see if the model will learn
# # to always traverse rightward

import numpy as np
import matplotlib.pyplot as plt

num_episodes =  1000
epsilon = 0.9
discount = 0.9 # Change to 1 to simplify Q-value results

state_rewards = [0]*20
state_rewards[0] = -20
state_rewards[-1] = 20

final_state = [False]*20
final_state[0] = True
final_state[-1] = True
Q_values = [[[0.0, 0.0] for a in range(len(state_rewards))] for q in range(num_episodes+1)]

def select_epsilon_greedy_action(epsilon, state, episode):
  """Take random action with probability epsilon, else take best action."""
  result = np.random.uniform()
  if result < epsilon:
    return np.random.randint(0, 2) # Random action (left or right)
  else:
    return np.argmax(Q_values[episode][state]) # Greedy action for state
  
def apply_action(state, action):
    """Applies the selected action and get reward and next state."""
    if action == 0: # action 0 moves us left
        next_state = state-1
    else: # action 1 moves us right
        next_state = state+1
    return state_rewards[next_state], next_state


for episode in range(1,num_episodes+1):
  initial_state = 10 # State in the middle
  state = initial_state # state reflects the current ststae and can range between 0 and 5 
  for i in range(len(Q_values[episode])):
        for j in range(len(Q_values[episode][i])):
            Q_values[episode][i][j] = Q_values[episode-1][i][j]
  while not final_state[state]: # Run until the end of the episode
    # Select action
    action = select_epsilon_greedy_action(epsilon, state, episode-1)
    reward, next_state = apply_action(state, action)
    # Improve Q-values with Bellman Equation    
    if final_state[next_state]:
      Q_values[episode][state][action] = reward
    else:
      Q_values[episode][state][action] = reward + discount * max(Q_values[episode][next_state])
    Q_values[episode][state][action] -= 1 # this is the cost incured of taking a step
    state = next_state

plt.plot([[q[1] for q in c] for c in Q_values])
plt.show()

# Print Q-values to see if action right is always better than action left
# except for states 0 and 6, which are terminal states and you cannot take
# any action from them, so it does not matter.
action_dict = {0:'left', 1:'right'}
state = 0
for Q_vals in Q_values[-1]:
  print('Best action for state {} is {}'.format(state, 
                                             action_dict[np.argmax(Q_vals)]))
  state += 1

print(Q_values[1])
print(Q_values[-1])
