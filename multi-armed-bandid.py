import numpy as np
import matplotlib.pyplot as plt

def pull_bandit_arm(bandits, bandit_number):
  """
  Pull arm in position bandit_number and return the obtained reward.
  """
  result = np.random.uniform()
  return int(result <= bandits[bandit_number])


def take_epsilon_greedy_action(epsilon, average_rewards):
  """
    Take random action with probability epsilon, else take best action.
  """
  result = np.random.uniform()
  if result < epsilon:
    return np.random.randint(0, len(average_rewards)) # Random action
  else:
    return np.argmax(average_rewards) # Greedy action
  

# Probability of success of each bandit
bandits = [0.1, 0.3, 0.05, 0.55, 0.4]
num_iterations = 10000
epsilon = 0.7

# Store info to know which one is the best action in each moment
total_rewards = [0 for _ in range(len(bandits))]
total_attempts = [0 for _ in range(len(bandits))]
average_rewards = [[0.0 for _ in range(len(bandits))] for q in range(num_iterations+1)]
print(np.shape(average_rewards))

for iteration in range(1,num_iterations+1):
  action = take_epsilon_greedy_action(epsilon, average_rewards[iteration-1]) # select an action, wither explorative or explitative
  reward = pull_bandit_arm(bandits, action)
  
  # Store result
  total_rewards[action] += reward
  total_attempts[action] += 1
  average_rewards[iteration][:] = average_rewards[iteration-1][:]
  average_rewards[iteration][action] = total_rewards[action] / float(total_attempts[action])
  
  if iteration % 100 == 0:
    print('Average reward for bandits in iteration {} is {}'.format(iteration,
                                  ['{:.2f}'.format(elem) for elem in average_rewards[iteration]]))

for q in range(5):
    plt.plot([average_rewards[a][q] for a in range(1000)])

for q in range(5):
  plt.plot([0,1000], 2*[bandits[q]], color = 'black')

plt.legend([str(q) for q in [0,1,2,3,4]])
plt.show()

# Print results
best_bandit = np.argmax(average_rewards[-1][:])
print('\nBest bandit is {} with an average observed reward of {:.4f}'
      .format(best_bandit, average_rewards[-1][best_bandit]))
print('Total observed reward in the {} episodes has been {}'
      .format(num_iterations, sum(total_rewards)))