import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from env import Mansion
import matplotlib.pyplot as plt

# Define the neural network model
class LSTM_DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_DQN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        q_values = self.fc(lstm_out[:, -1, :])  # Select the last time step
        return q_values

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

def process_state(state):
    state_tensor = torch.tensor(state, dtype=torch.float32)
    return state_tensor

# Hyperparameters
learning_rate = 0.9
epsilon = 0.1
gamma = 0.99
num_episodes = 50

# Initialize the Gym environment
env = Mansion()
episode_rewards = []
iteration = 0

# Calculate the state size
state_size = 0

for space_name, space in env.observation_space.spaces.items():
    if isinstance(space, gym.spaces.Dict):
        pass
    elif isinstance(space, gym.spaces.Tuple):
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

print("Action Space Size:", action_size)

hidden_size = 64
q_network = LSTM_DQN(state_size, hidden_size, action_size)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)



# Training loop
for episode in range(num_episodes):
    state = env.reset()
    state = flatten_dict_values(state)
    done = False
    total_reward = 0
    count = 0

    while not done:
        if random.uniform(0, 1) < epsilon:
            next_valid_action = env.list_next_action()
            action = next_valid_action[random.randint(0, len(next_valid_action) - 1)]
        else:
            with torch.no_grad():
                state_tensor = process_state(state)
                state_tensor = state_tensor.unsqueeze(0)
                state_tensor = state_tensor.unsqueeze(1)
                q_values = q_network(state_tensor)
                q_values = q_values.squeeze()
                valid_actions = env.list_next_action()
                valid_q_values = q_values[valid_actions]
                action = valid_actions[torch.argmax(valid_q_values).item()]

        state_tensor = process_state(state)
        next_state, reward, done, _ = env.step(action)

        if not done:
            next_state = flatten_dict_values(next_state)
            next_state_tensor = process_state(next_state)
            next_valid_action = env.list_next_action()
        else:
            next_state_tensor = None
            next_valid_action = []

        # Update the Q-value using the Bellman equation
        with torch.no_grad():
            state_tensor = process_state(state)
            state_tensor = state_tensor.unsqueeze(0)
            state_tensor = state_tensor.unsqueeze(1)
            q_values = q_network(state_tensor)
            q_values = q_values.squeeze()
            valid_q_values = q_values[next_valid_action]

            if len(valid_q_values) > 0:
                q_target = reward + gamma * torch.max(valid_q_values)
            else:
                q_target = reward

        if next_state_tensor is not None:
            q_value = q_network(state_tensor)
            q_target = q_target.expand_as(q_value)
            loss = nn.MSELoss()(q_value, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state
        total_reward += reward
        count += 1
        env.render()

        # Handle count == 550 condition
        if count == 550:
            done = True

    print(f"Episode {episode}, Total Reward: {total_reward}")
    episode_rewards.append(total_reward)

# Save the trained model
torch.save(q_network.state_dict(), 'mansion_qnetwork.pth')
plt.plot(range(len(episode_rewards)), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward vs. Episode')
plt.show()
