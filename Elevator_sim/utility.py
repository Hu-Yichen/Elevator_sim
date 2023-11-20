import gym
import torch
import numpy as np
from env import Mansion
import matplotlib.pyplot as plt
from DQN import QNetwork


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


# Hyperparameters
learning_rate = 0.001
epsilon = 0.2
gamma = 0.99
num_episodes = 10

# Initialize the Gym environment
env = Mansion()
episode_rewards = []
iteration = 0

# Calculate the state size
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
state = env.reset()
# print("Origional State:", state)
# print("Transformed State:", flatten_dict_values(state))

action_size = env.action_space.n

print("action space Size:", action_size)


# Initialize the Q-network and optimizer
q_network = QNetwork(state_size, action_size)
# Load the trained model parameters
q_network.load_state_dict(torch.load('elevator_qnetwork.pth'))
q_network.eval()  # Set the model to evaluation mode (important for dropout and batch normalization)

# state = env.reset()
# state = flatten_dict_values(state)
# print("Length of the state is:", len(state))
# print("Observation State:", state)
# print("Observation State(tensor):", torch.tensor(state, dtype=torch.float32))
# with torch.no_grad():
#     q_values = q_network(torch.tensor(state, dtype=torch.float32))
#     print("Q values:", q_values)
#     q_values = q_values.squeeze()
#     valid_actions = env.list_next_action()
#     print("All valid actions:", valid_actions)
#     valid_q_values = q_values[valid_actions]
#     print("Q values for valid actions:", valid_q_values)
#     action = valid_actions[torch.argmax(valid_q_values).item()]
#     print("Best action is:", action)
# next_state, reward, done, _ = env.step(action)
# with torch.no_grad():
#     next_state = flatten_dict_values(next_state)
#     q_target = reward + gamma * torch.max(q_network(torch.tensor(next_state, dtype=torch.float32)))
#     print("Q target:", q_target)
# q_value = q_network(torch.tensor(state, dtype=torch.float32))
# print("Q value:", q_value)
# Q_s_a = q_value[action]
# print("Q[s,a]:", Q_s_a)
# loss = nn.MSELoss()(Q_s_a, q_target)
# print("Loss:", loss)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()


#################### Training loop #########################
for episode in range(num_episodes):
    state = env.reset()
    state = flatten_dict_values(state)
    done = False
    total_reward = 0
    count = 0
    epsilon = 0.2


    while not done:
        with torch.no_grad():
            q_values = q_network(torch.tensor(state, dtype=torch.float32))
            q_values = q_values.squeeze()
            valid_actions = env.list_next_action()  # Get a list of valid actions
            valid_q_values = q_values[valid_actions]   # Extract Q-values for valid actions
            action = valid_actions[torch.argmax(valid_q_values).item()]


        next_state, reward, done, _ = env.step(action)
        env.render()

        # Update the Q-value using the Bellman equation
        next_state = flatten_dict_values(next_state)

        state = next_state
        total_reward += reward
        count += 1
        # env.render()
        if count == 500:
            done = True

    print(f"Episode {episode}, Total Reward: {total_reward}")
    episode_rewards.append(total_reward)
    iteration += 1

# Save the trained model
torch.save(q_network.state_dict(), 'elevator_qnetwork.pth')
plt.plot(range(iteration), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward vs. Episode')
plt.show()