# We show a simple example to start LiftSim here
# import gymnasium as gym
# from gymnasium import spaces
import gym
from gym import spaces
import numpy as np
from collections import namedtuple
from collections import deque
from person import Passenger_Generator
from itertools import product
from render import Render
import random
import math


NoDisplay = False

class Elevator(object):
    '''
    '''
    def __init__(self, elenum, max_floor):
        self.elenum = elenum
        self.ele_cur_position = [1] * elenum
        self.max_floor = max_floor
        self.cur_loading = [0] * elenum
        self.max_loading = 13
        self.time_per_floor = 2
        self.door_opening_time = 2
        self.door_closing_time = 2
        self.energy_consumption_to_open_door_per_time = 1
        self.energy_consumption_to_close_door_per_time = 1
        self.energy_consumption_for_going_up_per_time = 1
        self.energy_consumption_for_going_down_per_time = 1
        self.energy_consumption_for_staying_still_per_time = 0
        self.cur_move_direction = [0] * elenum
        self.current_dispatch_target = [[] for _ in range(elenum)]
        self.dispatch_target_direction = [0] * elenum
        self.over_loading_alarm = [False] * elenum
        self.door_is_open = [False] * elenum
        self.door_is_close = [True] * elenum
        self.is_opening = [False] * elenum
        self.is_closing = [False] * elenum
        self.is_ready_to_load = [False] * elenum
        self.time_need_to_close_door = [0] * elenum
        self.time_need_to_open_door = [0] * elenum
        self.staying_time = [0]*elenum

    # return the observation space of the elevator
    def ele_state(self):

        target = [[0 for _ in range(self.max_floor)] for _ in range(self.elenum)]
        for ele in range(self.elenum):
            for i in self.current_dispatch_target[ele]:
                target[ele][i-1] = 1

        over_loading_alarm = [0]*self.elenum
        door_is_open = [0]*self.elenum
        door_is_close = [0]*self.elenum
        is_ready_to_load = [0]*self.elenum
        is_opening = [0]*self.elenum
        is_closing = [0]*self.elenum
        for ele in range(self.elenum):
            if self.over_loading_alarm[ele]:
                over_loading_alarm[ele] = 1
            if self.door_is_open[ele]:
                door_is_open[ele] = 1
            if self.door_is_close[ele]:
                door_is_close[ele] = 1
            if self.is_ready_to_load[ele]:
                is_ready_to_load[ele] = 1
            if self.is_opening[ele]:
                is_opening[ele] = 1
            if self.is_closing[ele]:
                is_closing[ele] = 1


        state_dictionary = {
        "ele_cur_position": self.ele_cur_position,
        "cur_move_direction": self.cur_move_direction,
        "dispatch_target_direction": self.dispatch_target_direction,
        "current_loading": self.cur_loading,
        "over_loading_alarm": over_loading_alarm,
        "door_is_open": door_is_open,
        "door_is_close": door_is_close,
        "is_ready_to_load": is_ready_to_load,
        "is_opening": is_opening,
        "is_closing": is_closing,
        "time_need_to_close_door": self.time_need_to_close_door,
        "time_need_to_open_door": self.time_need_to_open_door,
        "target": target
        # "current dispatch_target": self.current_dispatch_target
        }
    
        return state_dictionary

    # run elevator one step with action
    def run_elevator(self, action, wait_upward_persons_queue, wait_downward_persons_queue):

        # for i in range(self.elenum):
        #     if action[i]==0:
        #         self.staying_time[i] += 1
        #         if self.staying_time[i] >=100:
        #             print(self.ele_state())
        #     else:
        #         self.staying_time[i] = 0

        energy_consumption = 0

        self.cur_move_direction = action.copy() # update current direction same as the action

        for i in range(self.elenum):
            self.ele_cur_position[i] += action[i] / self.time_per_floor # update elevators position after action

        for i in action: # compute energy consumption with action
            if(i == 1):
                energy_consumption += self.energy_consumption_for_going_up_per_time
            if(i == -1):
                energy_consumption += self.energy_consumption_for_going_down_per_time
            if(i == 0):
                energy_consumption += self.energy_consumption_for_staying_still_per_time


        # update dispatch_target_direction according to action and location (recording directions for stopping elevators)
        for i in range(self.elenum):
            if self.current_dispatch_target[i]:
                if(action[i] != 0 and self.ele_cur_position[i] != 1 and self.ele_cur_position[i] != self.max_floor):
                    self.dispatch_target_direction[i] = action[i]
                elif(self.ele_cur_position[i] == 1):
                    self.dispatch_target_direction[i] = 1
                elif(self.ele_cur_position[i] ==self.max_floor):
                    self.dispatch_target_direction[i] = -1

        # Decision Tree
        for i in range(self.elenum):
            if(self.door_is_open[i] == True): # if door is open
                if not self.is_integer(self.ele_cur_position[i]):
                    raise ValueError('The door have to be closed when the elevator is not on the floor')
                if(self.is_ready_to_load[i] == True): # if is ready to load
                ### --- Decision making when door is open and ready to load  --- ###
                    if self.dispatch_target_direction[i] == 1: # if elevator is going up
                        if(self.is_closing[i] == True): # if the door is going to close
                            if wait_upward_persons_queue[int(self.ele_cur_position[i])-1]: # if have passenger wait outside
                                energy_consumption += 1 # assume the energy used for reopen the door is 1
                                self.is_closing[i] = False
                            elif not wait_upward_persons_queue[int(self.ele_cur_position[i])-1]: # if there is no new arrival people
                                self.time_need_to_close_door[i] -= 1
                                energy_consumption += self.energy_consumption_to_close_door_per_time
                                if(self.time_need_to_close_door[i] == 0):
                                    self.door_is_open[i] = False
                                    self.door_is_close[i] = True
                                    self.is_closing[i] = False
                                    self.is_ready_to_load[i] = False
                        elif(self.is_closing[i] == False):
                            if wait_upward_persons_queue[int(self.ele_cur_position[i])-1]: # if have passenger wait outside
                                self.current_dispatch_target[i].append(wait_upward_persons_queue[int(self.ele_cur_position[i])-1][0].destination)
                                self.cur_loading[i] += 1
                                wait_upward_persons_queue[int(self.ele_cur_position[i])-1].popleft()
                                if(self.cur_loading[i]== self.max_loading): # check if is overloading
                                    self.over_loading_alarm[i] = True
                                    self.is_ready_to_load[i] = False
                                    self.is_closing[i] = True
                                    self.time_need_to_close_door[i] = self.door_closing_time
                                    continue
                                if wait_upward_persons_queue[int(self.ele_cur_position[i])-1]: # if there are still passenger waiting out side
                                    continue
                                elif not wait_upward_persons_queue[int(self.ele_cur_position[i])-1]:
                                    self.is_closing[i] = True
                                    self.time_need_to_close_door[i] = self.door_closing_time
                            elif not wait_upward_persons_queue[int(self.ele_cur_position[i])-1]: # if no one waiting outside
                                self.is_closing[i] = True
                                self.time_need_to_close_door[i] = self.door_closing_time
                    elif self.dispatch_target_direction[i] == -1 : # if elevator is going down
                        if(self.is_closing[i] == True): # if the door is going to close
                            if(wait_downward_persons_queue[int(self.ele_cur_position[i])-1]): # if have passenger wait outside
                                energy_consumption += 1 # assume the energy used for reopen the door is 1
                                self.is_closing[i] = False
                            elif not wait_downward_persons_queue[int(self.ele_cur_position[i])-1]: # if there is no new arrival people
                                self.time_need_to_close_door[i] -= 1
                                energy_consumption += self.energy_consumption_to_close_door_per_time
                                if(self.time_need_to_close_door[i] == 0):
                                    self.door_is_open[i] = False
                                    self.door_is_close[i] = True
                                    self.is_closing[i] = False
                                    self.is_ready_to_load[i] = False
                        elif(self.is_closing[i] == False):
                            if(wait_downward_persons_queue[int(self.ele_cur_position[i])-1]): # if have passenger wait outside
                                self.current_dispatch_target[i].append(wait_downward_persons_queue[int(self.ele_cur_position[i])-1][0].destination)
                                self.cur_loading[i] += 1
                                wait_downward_persons_queue[int(self.ele_cur_position[i]-1)].popleft()
                                if(self.cur_loading[i]== self.max_loading): # check if is overloading
                                    self.over_loading_alarm[i] = True
                                    self.is_ready_to_load[i] = False
                                    self.is_closing[i] = True
                                    self.time_need_to_close_door[i] = self.door_closing_time
                                    continue
                                if(wait_downward_persons_queue[int(self.ele_cur_position[i])-1]): # if there are still passenger waiting out side
                                    continue
                                elif not wait_downward_persons_queue[int(self.ele_cur_position[i])-1]:
                                    self.is_closing[i] = True
                                    self.time_need_to_close_door[i] = self.door_closing_time
                            elif not wait_downward_persons_queue[int(self.ele_cur_position[i])-1]: # if no one waiting outside
                                self.is_closing[i] = True
                                self.time_need_to_close_door[i] = self.door_closing_time
                    elif self.dispatch_target_direction[i] == 0:
                        if(self.is_closing[i] == True): # if the door is going to close
                            if(wait_downward_persons_queue[int(self.ele_cur_position[i])-1]): # if have passenger wait outside
                                energy_consumption += 1 # assume the energy used for reopen the door is 1
                                self.is_closing[i] = False
                            elif(wait_upward_persons_queue[int(self.ele_cur_position[i])-1]): # if have passenger wait outside
                                energy_consumption += 1 # assume the energy used for reopen the door is 1
                                self.is_closing[i] = False
                            elif not wait_upward_persons_queue[int(self.ele_cur_position[i])-1] and not wait_downward_persons_queue[int(self.ele_cur_position[i])-1]: # if there is no new arrival people
                                self.time_need_to_close_door[i] -= 1
                                energy_consumption += self.energy_consumption_to_close_door_per_time
                                if(self.time_need_to_close_door[i] == 0):
                                    self.door_is_open[i] = False
                                    self.door_is_close[i] = True
                                    self.is_closing[i] = False
                                    self.is_ready_to_load[i] = False
                        elif(self.is_closing[i] == False):
                            if wait_downward_persons_queue[int(self.ele_cur_position[i])-1]: # if have passenger wait outside
                                self.current_dispatch_target[i].append(wait_downward_persons_queue[int(self.ele_cur_position[i])-1][0].destination)
                                self.cur_loading[i] += 1
                                self.dispatch_target_direction[i] = -1
                                wait_downward_persons_queue[int(self.ele_cur_position[i])-1].popleft()
                                if(wait_downward_persons_queue[int(self.ele_cur_position[i])-1]): # if there are still passenger waiting out side
                                    continue
                                elif not wait_downward_persons_queue[int(self.ele_cur_position[i])-1]:
                                    self.is_closing[i] = True
                                    self.time_need_to_close_door[i] = self.door_closing_time
                            if(wait_upward_persons_queue[int(self.ele_cur_position[i])-1]): # if have passenger wait outside
                                self.current_dispatch_target[i].append(wait_upward_persons_queue[int(self.ele_cur_position[i])-1][0].destination)
                                self.cur_loading[i] += 1
                                self.dispatch_target_direction[i] = 1
                                wait_upward_persons_queue[int(self.ele_cur_position[i])-1].popleft()
                                if(wait_upward_persons_queue[int(self.ele_cur_position[i])-1]): # if there are still passenger waiting out side
                                    continue
                                elif not wait_upward_persons_queue[int(self.ele_cur_position[i])-1]:
                                    self.is_closing[i] = True
                                    self.time_need_to_close_door[i] = self.door_closing_time
                            elif not wait_downward_persons_queue[int(self.ele_cur_position[i])-1] and not wait_upward_persons_queue[int(self.ele_cur_position[i])-1]: # if no one waiting outside
                                self.is_closing[i] = True
                                self.time_need_to_close_door[i] = self.door_closing_time

                ### --- Decision making when door is open and able to load passengers END   END   END --- ###

                ### --- Decision making when door is open and unable to load passengers --- ###
                elif(self.is_ready_to_load[i] == False): # three possibilities for the elevator when it is unable to load: 1. passenger get off 2. overload 3.open for loading
                    have_passenger_arrive, position_in_ele = self.get_passenger_arrive(i)
                    if(have_passenger_arrive == True): # if some passengers in elevator arrive and get off in the floor
                        del self.current_dispatch_target[i][position_in_ele]
                        self.cur_loading[i] -= 1
                        have_passenger_arrive, _ = self.get_passenger_arrive(i)
                        if(have_passenger_arrive == False):
                            self.is_ready_to_load[i] = True
                        if(self.over_loading_alarm[i] == True):
                            self.over_loading_alarm[i] = False
                        if not self.current_dispatch_target[i]:
                            self.dispatch_target_direction[i] = 0
                            self.is_ready_to_load[i] = True
                    elif(have_passenger_arrive == False): # if no passenger in elevator get off in the floor
                        if(self.over_loading_alarm[i] == True): # if overload
                            self.time_need_to_close_door[i] -= 1
                            energy_consumption += self.energy_consumption_to_close_door_per_time
                            if(self.time_need_to_close_door[i] == 0):
                                self.door_is_open[i] = False
                                self.door_is_close[i] = True
                                self.is_closing[i] = False
                        elif(self.over_loading_alarm[i] == False): # open for loading
                            self.is_ready_to_load[i] = True
                ### --- Decision making when door is open and unable to load passengers END   END   END --- ###


            ### --- if door is closed --- ###
            elif(self.door_is_open[i] == False): # if door is closed
                if self.current_dispatch_target[i]: # if someone is in elevator
                    if self.is_opening[i] == True: # if door is opening
                        self.time_need_to_open_door[i] -= 1
                        energy_consumption +=  self.energy_consumption_to_open_door_per_time
                        if(self.time_need_to_open_door[i] == 0):
                            self.door_is_open[i] = True
                            self.door_is_close[i] = False
                            self.is_opening[i] = False
                    elif(self.is_opening[i] == False): # if door is not opening
                        have_request = self.get_request(i, wait_upward_persons_queue, wait_downward_persons_queue)
                        have_passenger_arrive, _ = self.get_passenger_arrive(i)
                        if(have_passenger_arrive == True): # if door is not opening and passengers in elevator have arrived
                            self.is_opening[i] = True
                            self.time_need_to_open_door[i] = self.door_opening_time
                            continue
                        if(have_request == True): # if door is not opening and have recieved request from the floor
                            if(self.over_loading_alarm[i] == True):
                                print("The lift is over loaded, please wait for the next one")
                            elif(self.over_loading_alarm[i] == False):
                                if self.cur_move_direction[i] == 0:
                                    self.is_opening[i] = True
                                    self.time_need_to_open_door[i] = self.door_opening_time
                                else:
                                    continue
                        else: # no request and no arrive, elevator is still running
                            continue
                elif not self.current_dispatch_target[i]: # no one is in the elevator
                    if not self.is_opening[i]:
                        have_request = self.get_request(i, wait_upward_persons_queue, wait_downward_persons_queue)
                        if(have_request == True):
                            self.is_opening[i] = True
                            self.time_need_to_open_door[i] = self.door_opening_time
                        else: # no request and no arrive and no destination, randomly choose a cation
                            continue
                    elif self.is_opening[i]:
                        self.time_need_to_open_door[i] -= 1
                        energy_consumption += self.energy_consumption_to_open_door_per_time
                        if self.time_need_to_open_door[i] == 0:
                            self.door_is_open[i] = True
                            self.door_is_close[i] = False
                            self.is_opening[i] = False
                            self.is_ready_to_load[i] = True

        return energy_consumption, wait_upward_persons_queue, wait_downward_persons_queue


    # return if any passengers inside elevator arrive at current floor
    def get_passenger_arrive(self, ele_index):
        have_passenger_arrive = False
        position_in_ele = 0
        for index, destination in enumerate(self.current_dispatch_target[ele_index]):
            if(destination == self.ele_cur_position[ele_index]):
                have_passenger_arrive = True
                position_in_ele = index
                break
        return have_passenger_arrive, position_in_ele


    # return if any passenger outside  are waiting at current floor
    def get_request(self, ele_index, wait_upward_persons_queue, wait_downward_persons_queue):
        have_request = False
        if self.is_integer(self.ele_cur_position[ele_index]):
            current_position = int(self.ele_cur_position[ele_index])  # Convert to integer
            if(self.dispatch_target_direction[ele_index] == 1 or self.dispatch_target_direction[ele_index] == 0):
                if wait_upward_persons_queue[current_position-1]:
                    have_request = True
            if(self.dispatch_target_direction[ele_index] == -1 or self.dispatch_target_direction[ele_index] == 0):
                if wait_downward_persons_queue[current_position-1]:
                    have_request = True
        return have_request


    def is_integer(self, num):
        converted = int(num)
        return num == converted



class Mansion(gym.Env):
    def __init__(self, elenum = 3, number_of_floors = 10, passenger_generate_type = 0):
        if(elenum > 8 or elenum <3):
            raise NotImplementedError("The number of element illegal, please enter a value between 3 to 8")
        if(number_of_floors > 30 or number_of_floors <8):
            raise NotImplementedError("The number of floors illegal, please enter a value between 8 to 30")
        assert elenum > 0, "elenum must be a positive integer"
        assert number_of_floors > 0, "number_of_floors must be a positive integer"
        self.number_of_floors = number_of_floors
        self.floor_height = 4
        self.elenum = elenum
        # self.max_loading = 13
        self.time_step = 0
        self.dt = 1
        self.person_entering_time = 1
        self.time_per_floor = 2
        self.viewer = None
        self.passenger_generate_type = passenger_generate_type
        self.mansion_wait_time = [[0 for i in range(number_of_floors)] for j in range(2)]
        ele_action = [[-1, 0, 1] for i in range(self.elenum)]
        action_tuple = list(product(*ele_action))
        self.action = [list(combination) for combination in action_tuple]
        # print(self.action_space)
        self.action_space = spaces.Discrete(3**elenum)
        self.observation_space = spaces.Dict({
            'ele_cur_position'             : spaces.Box(low=1, high=number_of_floors, shape=(1, elenum), dtype=float),
            'cur_move_direction'           : spaces.Box(low=-1, high=1, shape=(1, elenum), dtype=int),
            'dispatch_target_direction'    : spaces.Box(low=-1, high=1, shape=(1, elenum), dtype=int),
            'current_loading'              : spaces.Box(low=0, high=np.inf, shape=(1, elenum), dtype=int),
            'over_loading_alarm'           : spaces.MultiBinary(elenum),
            'door_is_open'                 : spaces.MultiBinary(elenum),
            'door_is_close'                : spaces.MultiBinary(elenum),
            'is_ready_to_load'             : spaces.MultiBinary(elenum),
            'is_opening'                   : spaces.MultiBinary(elenum),
            'is_closing'                   : spaces.MultiBinary(elenum),
            'time_need_to_close_door'      : spaces.MultiDiscrete([3] * elenum),
            'time_need_to_open_door'       : spaces.MultiDiscrete([3] * elenum),
            'target'                       : spaces.Tuple([
                                             spaces.MultiDiscrete([2] * number_of_floors) for _ in range(elenum)
                                             ]),
            'mansion_wait_time'            : spaces.Tuple([
                                             spaces.Box(low=0, high=np.inf, shape=(1, number_of_floors), dtype=int),
                                             spaces.Box(low=0, high=np.inf, shape=(1, number_of_floors), dtype=int)
                                             ])
            })


    def reset(self):
        self.time_step = 0 # initialize current time to 0
        self.elevator = Elevator(elenum = self.elenum, max_floor = self.number_of_floors) # create elevators
        self.passenger_generator = Passenger_Generator(generate_type = self.passenger_generate_type, max_floor = self.number_of_floors) # create passenger generator
        self.button_state = [[False, False] for i in range(self.number_of_floors)] # initialize all button into deactive
        self._wait_upward_persons_queue = [ deque() for i in range(self.number_of_floors)] # initialize passengers who want to go up to empty
        self._wait_downward_persons_queue = [ deque() for i in range(self.number_of_floors)]# initialize passengers who want to go down to empty

        # for i in range(self.elenum):
        #     if self.elevator.ele_cur_position[i] == 1:
        #         self.elevator.dispatch_target_direction[i] = 1
        #     elif self.elevator.ele_cur_position[i] == self.number_of_floors:
        #         self.elevator.dispatch_target_direction[i] = -1

        return self._mansion_observe()

    # return observation space
    def _mansion_observe(self):
        ele_state = self.elevator.ele_state()
        ele_state["mansion_wait_time"] = self.mansion_wait_time


        observation_state = ele_state

        return observation_state
    

    def step(self, action_index):

        action = self.index2act(action_index)

        self.time_step += 1

        passenger_list = self.passenger_generator.generate_passenger() # generate passengers


        for passenger in passenger_list: # add new passengers to waiting list
            if passenger.current_position < passenger.destination:
                self._wait_upward_persons_queue[(passenger.current_position)-1].append(passenger)
            elif passenger.current_position > passenger.destination:
                self._wait_downward_persons_queue[(passenger.current_position)-1].append(passenger)

        # elevator move one step
        energy_consumption, self._wait_upward_persons_queue, self._wait_downward_persons_queue = self.elevator.run_elevator(
            action, self._wait_upward_persons_queue, self._wait_downward_persons_queue)

        # update waiting time for each passengers
        for i in range(self.number_of_floors):
            for passenger in self._wait_upward_persons_queue[i]:
                passenger.waiting_time += 1
            for passenger in self._wait_downward_persons_queue[i]:
                passenger.waiting_time += 1    


        # compute total reward
        reward = self.compute_reward(energy_consumption, self._wait_upward_persons_queue, self._wait_downward_persons_queue)

        # update button state
        for floor in range(self.number_of_floors):
            if(len(self._wait_upward_persons_queue[floor]) > 0):
                self.button_state[floor][0] = True
            else:
                self.button_state[floor][0] = False

            if(len(self._wait_downward_persons_queue[floor]) > 0):
                self.button_state[floor][1] = True
            else:
                self.button_state[floor][1] = False

        # update mansion wait time
        for floor in range(self.number_of_floors):
            if(len(self._wait_upward_persons_queue[floor]) > 0):
                self.mansion_wait_time[0][floor] = self._wait_upward_persons_queue[floor][0].waiting_time
            else:
                 self.mansion_wait_time[0][floor] = 0
            if(len(self._wait_downward_persons_queue[floor]) > 0):
                self.mansion_wait_time[1][floor] = self._wait_downward_persons_queue[floor][0].waiting_time
            else:
                 self.mansion_wait_time[1][floor] = 0

        # compute next possible actions
        next_possible_action = []
        next_possible_action = self.list_next_action()

        return self._mansion_observe(), reward, False, next_possible_action



    def compute_reward(self, energy, wait_upward_persons_queue, wait_downward_persons_queue):
        waiting_time = []
        for i in range(self.number_of_floors):
            for passenger in wait_upward_persons_queue[i]:
                waiting_time.append(passenger.waiting_time)
        for i in range(self.number_of_floors):
            for passenger in wait_downward_persons_queue[i]:
                waiting_time.append(passenger.waiting_time)
        time_reward = 0
        energy_reward = 1.5*energy
        for i in waiting_time:
            time_reward += -2.5*(i) + 50
        reward = time_reward - energy_reward
        return reward
    
    # def compute_reward(self, energy, wait_upward_persons_queue, wait_downward_persons_queue):
    #     cumulative_waiting_time = 0
    #     for i in range(self.number_of_floors):
    #         cumulative_waiting_time -= len(wait_upward_persons_queue)
    #         cumulative_waiting_time -= len(wait_downward_persons_queue)
    #     # cumulative_waiting_time += loaded_person_num
    #     energy_reward = 5*energy
    #     reward = cumulative_waiting_time - energy_reward
    #     return reward

    def list_next_action(self):
        ele_next_action = [[] for i in range(self.elenum)]
        for i in range(self.elenum): # iterate all lifts
            if self.elevator.door_is_open[i]: # if the door is open, then the lift have to stay still
                ele_next_action[i].append(0)
            elif not self.elevator.door_is_open[i]: 
                if self.elevator.is_opening[i]: # if the door is opening, then the lift have to stay still
                    ele_next_action[i].append(0)
                elif not self.elevator.is_opening[i]:
                    if self.elevator.current_dispatch_target[i]:
                        if self.get_passenger_arrive(i): # if have passenger arrive, stay
                            ele_next_action[i].append(0)
                        elif not self.get_passenger_arrive(i):
                            if self.elevator.dispatch_target_direction[i] == 1:
                                if self.get_request(i):
                                    if self.elevator.over_loading_alarm[i]: # if overloading alarm, continue moving
                                        ele_next_action[i].append(1)
                                    elif not self.elevator.over_loading_alarm[i]: # if not alarm, anwser requeset outside, stay
                                        ele_next_action[i].append(0)
                                        ele_next_action[i].append(1)
                                elif not self.get_request(i): # if no passenger arrive and no request
                                    ele_next_action[i].append(1)
                            elif self.elevator.dispatch_target_direction[i] == -1:
                                if self.get_request(i):
                                    if self.elevator.over_loading_alarm[i]: # if overloading alarm, continue moving
                                        ele_next_action[i].append(-1)
                                    elif not self.elevator.over_loading_alarm[i]: # if not alarm, anwser requeset outside, stay
                                        ele_next_action[i].append(0)
                                        ele_next_action[i].append(-1)
                                elif not self.get_request(i): # if no passenger arrive and no request
                                    ele_next_action[i].append(-1)
                    elif not self.elevator.current_dispatch_target[i]:
                        if self.is_integer(self.elevator.ele_cur_position[i]):
                            # current_position = int(self.elevator.ele_cur_position[i])
                            # if self.button_state[current_position-1][0] or self.button_state[current_position-1][1]:
                            #     if self.elevator.dispatch_target_direction[i] == 0: # if door is closed, no one in lift, have requeset outside, current moving direction is 0, then stay
                            #         ele_next_action[i].append(0)
                            #         ele_next_action[i].append(1)
                            #         ele_next_action[i].append(-1)
                                # elif self.elevator.dispatch_target_direction[i] == 1: 
                                #     print(1)
                                #     if self.button_state[current_position-1][0]: # if door is closed, no one in lift, have requeset outside, same direction with moving direction, then stay
                                #         ele_next_action[i].append(0)
                                #     elif self.button_state[current_position-1][1]:
                                #         if self.have_prioritized_request(i): # if door is closed, no one in lift, have requeset outside, different direction with moving direction, but exist prioritized request then continue moving
                                #             ele_next_action[i].append(1)
                                #         elif not self.have_prioritized_request(i): # if door is closed, no one in lift, have requeset outside, different direction with moving direction, no prioritized request then continue moving
                                #             ele_next_action[i].append(0)
                                # elif self.elevator.dispatch_target_direction[i] == -1:
                                #     print(2)
                                #     if self.button_state[current_position-1][1]: # if door is closed, no one in lift, have requeset outside, same direction with moving direction, then stay
                                #         ele_next_action[i].append(0)
                                #     elif self.button_state[current_position-1][0]:
                                #         if self.have_prioritized_request(i): # if door is closed, no one in lift, have requeset outside, different direction with moving direction, but exist prioritized request then continue moving
                                #             ele_next_action[i].append(-1)
                                #         elif not self.have_prioritized_request(i): # if door is closed, no one in lift, have requeset outside, different direction with moving direction, no prioritized request then continue moving
                                #             ele_next_action[i].append(0)  
                            # else:
                                # if self.elevator.ele_cur_position[i] == self.number_of_floors: # if no one in lift and no request and lift is at the top floor
                                #     ele_next_action[i].append(0)
                                #     ele_next_action[i].append(-1)
                                # elif self.elevator.ele_cur_position[i] == 1:                   # if no one in lift and no request and lift is at bottom
                                #     ele_next_action[i].append(0)
                                #     ele_next_action[i].append(1)
                                # else:                                                          # if no one in lift and no request and lift is not at the bottom or top floor
                                #     ele_next_action[i].append(0)
                                #     ele_next_action[i].append(1)
                                #     ele_next_action[i].append(-1)
                            if self.elevator.ele_cur_position[i] == self.number_of_floors: # if no one in lift and no request and lift is at the top floor
                                    ele_next_action[i].append(0)
                                    ele_next_action[i].append(-1)
                            elif self.elevator.ele_cur_position[i] == 1:                   # if no one in lift and no request and lift is at bottom
                                ele_next_action[i].append(0)
                                ele_next_action[i].append(1)
                            else:                                                          # if no one in lift and no request and lift is not at the bottom or top floor
                                ele_next_action[i].append(0)
                                ele_next_action[i].append(1)
                                ele_next_action[i].append(-1)
                        else:
                            # if self.elevator.ele_cur_position[i] == self.number_of_floors: # if no one in lift and no request and lift is at the top floor
                            #     ele_next_action[i].append(0)
                            #     ele_next_action[i].append(-1)
                            # elif self.elevator.ele_cur_position[i] == 1:                   # if no one in lift and no request and lift is at bottom
                            #     ele_next_action[i].append(0)
                            #     ele_next_action[i].append(1)
                            # else:                                                          # if no one in lift and no request and lift is not at the bottom or top floor
                            #     # ele_next_action[i].append(0)
                            #     ele_next_action[i].append(1)
                            #     ele_next_action[i].append(-1)
                            ele_next_action[i].append(1)
                            ele_next_action[i].append(-1)

        next_action_tuple = list(product(*ele_next_action))
        next_action_list = [list(combination) for combination in next_action_tuple]
        next_action_index = self.act2index(next_action_list)
        return next_action_index


    # return if someone have request from floors that have same action direction
    def have_prioritized_request(self, ele_index):
        have_request = False
        if self.elevator.dispatch_target_direction[ele_index] == 1:
            check_position = math.ceil(self.elevator.ele_cur_position[ele_index])
            for floor in range(check_position, self.number_of_floors):
                if self.button_state[floor-1][0] == True:
                    have_request = True
        elif self.elevator.dispatch_target_direction[ele_index] == -1:
            check_position = math.floor(self.elevator.ele_cur_position[ele_index])
            for floor in range(1, check_position):
                if self.button_state[floor-1][0] == True:
                    have_request = True
        return have_request

    # return if passenger arrive
    def get_passenger_arrive(self, ele_index):
        have_passenger_arrive = False
        for  destination in enumerate(self.elevator.current_dispatch_target[ele_index]):
            if(destination == self.elevator.ele_cur_position[ele_index]):
                have_passenger_arrive = True
                break
        return have_passenger_arrive

    # return if passenger outside have request 
    def get_request(self, ele_index):
        have_request = False
        if self.is_integer(self.elevator.ele_cur_position[ele_index]):
            current_position = int(self.elevator.ele_cur_position[ele_index])  # Convert to integer
            if(self.elevator.dispatch_target_direction[ele_index] == 1 or self.elevator.dispatch_target_direction[ele_index] == 0):
                if(self._wait_upward_persons_queue[current_position-1]):
                    have_request = True
            if(self.elevator.dispatch_target_direction[ele_index] == -1 or self.elevator.dispatch_target_direction[ele_index] == 0):
                if(self._wait_downward_persons_queue[current_position-1]):
                    have_request = True
        return have_request

    def is_integer(self, num):
        converted = int(num)
        return num == converted

    def render(self, mode="human"):
        if(mode != "human"):
            raise NotImplementedError("Only support human mode currently")
        if self.viewer is None:
            if NoDisplay:
                raise Exception('[Error] Cannot connect to display screen. \
                    \n\rYou are running the render() function on a manchine that does not have a display screen')
            self.viewer = Render(self)
        self.viewer.view()

    def index2act(self, index):
        return self.action[index]

    def act2index(self, list):
        index = [self.action.index(item) for item in list]
        return index



if __name__=='__main__':
    env = Mansion(elenum = 5, number_of_floors = 10)
    env.reset()
    # print("Observation state:")
    # print(env._mansion_observe())
    valid_actions = []

    for i in range(1000):
        env.render()
        valid_actions = env.list_next_action()
        next_action = valid_actions[random.randint(0,len(valid_actions)-1)]
        env.step(next_action)
        valid_actions = []
    print("Observation state:")
    print(env._mansion_observe())
