import gym
import torch
import torch.nn as nn
import numpy as np
from env import Mansion

# Assuming you have a Q-network class defined like this
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)
    # Your Q-network architecture here

def flatten_dict_values(d):
    flat_list = []
    for key, value in d.items():
        if isinstance(value, list):
            for i in value:
                if isinstance(i, (list, tuple, set)):
                    flat_list.extend(i)
                else:
                    flat_list.append(i)
        elif isinstance(value, dict):
            flat_list.extend(flatten_dict_values(value))
        else:
            flat_list.append(value)
    return flat_list

env = Mansion()  # Create your Gym environment
state = env.reset()  # Reset the environment to get the initial state
total_steps = 0  # To keep track of the number of steps taken
max_steps = 1000  # The desired number of steps
state = flatten_dict_values(state)
next_action = env.list_next_action()

state_size = 0

for space_name, space in env.observation_space.spaces.items():
    # if space is not None:
    if isinstance(space, gym.spaces.Dict):
        # For nested Dict spaces, you can further iterate or process them as needed
        pass
    elif isinstance(space, gym.spaces.Tuple):
        # Calculate the size of the Tuple
        tuple_size = 0
        for sub_space in space.spaces:
            if sub_space.shape is not None:
                tuple_size += int(np.prod(sub_space.shape))
        state_size += tuple_size
    else:
        if space.shape is not None:
            state_size += int(np.prod(space.shape))

print("State Size:", state_size)

action_size = env.action_space.n

print("action space Size:", action_size)

total_reward = 0

# Load your saved Q-network
q_network = QNetwork(state_size, action_size)
q_network.load_state_dict(torch.load('elevator_qnetwork.pth'))
q_network.eval()  # Set the Q-network to evaluation mode

while total_steps < max_steps:

    q_values = q_network(torch.tensor(state, dtype=torch.float32))
    q_values = q_values.squeeze()
    # action = torch.argmax(q_values).item()  # Exploit by selecting the action with the highest Q-value
    valid_actions = env.list_next_action()  # Get a list of valid actions
    valid_q_values = q_values[valid_actions]   # Extract Q-values for valid actions
    action = valid_actions[torch.argmax(valid_q_values).item()]  # Choose the action with the highest Q-value among valid actions

    # Take the selected action in the environment
    next_state, reward, done, _ = env.step(action)

    env.render()

    next_state = flatten_dict_values(next_state)

    # Update the state for the next iteration
    state = next_state

    # Increment the total step count
    total_steps += 1
    total_reward += reward
    if done:
        break

print(total_reward)
# You've now run the environment for 1000 steps using the Q-network
env.close()  # Close the environment when done
