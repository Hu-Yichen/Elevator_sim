from env import Mansion
import random

elevator_number = 3
floor_number = 10

# env = Mansion(elenum = elevator_number, number_of_floors = floor_number)
env = Mansion()

env.reset()
print("Observation state after initialization:")
print(env._mansion_observe())
valid_actions = []

for i in range(500):
    env.render()
    valid_actions = env.list_next_action()
    next_action = valid_actions[random.randint(0,len(valid_actions)-1)]
    env.step(next_action)
    valid_actions = []
print("Observation state after iteration:")
print(env._mansion_observe())
