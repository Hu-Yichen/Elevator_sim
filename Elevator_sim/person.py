import numpy as np

import random

class Passenger(object):

    def __init__(self, current_position, destination):
        self.current_position = current_position
        self.destination = destination
        self.waiting_time = 0

    def get_current_position(self):
        return self.current_position

    def get_destination(self):
        return self.destination

    def get_waiting_time(self):
        return self.waiting_time

class Passenger_Generator(object):

    def __init__(self, generate_type = 0, max_floor = 30):
        self.timer = 0
        self.generate_type = generate_type
        self.max_floor = max_floor
        self.passenger_list = self.generate_passenger()
        

    def generate_passenger(self):
        self.timer += 1
        noisy_passenger_num = random.choices([0, 1], weights=[0.96, 0.04], k=1)
        passenger_list = []

        if self.generate_type == 1:
            for i in range(noisy_passenger_num[0]):
                current_position = random.randint(1, self.max_floor)
                destination = random.randint(1,self.max_floor)
                while current_position == destination:
                    destination = random.randint(1, self.max_floor)
                passenger = Passenger(current_position, destination)
                passenger_list.append(passenger)

        if self.generate_type == 0:
            if self.timer == 5:
                for i in range(5):
                    passenger = Passenger(1+2*i, 1)
                    passenger_list.append(passenger)
            if self.timer == 75:
                passenger_list.append(Passenger(1, random.randint(2,10)))
                passenger_list.append(Passenger(1, random.randint(2,10))) 
                passenger_list.append(Passenger(1, random.randint(2,10))) 
                passenger_list.append(Passenger(1, random.randint(2,10)))
            if self.timer == 125:
                passenger_list.append(Passenger(1, random.randint(2,10)))
                passenger_list.append(Passenger(1, random.randint(2,10))) 
                passenger_list.append(Passenger(1, random.randint(2,10))) 
                passenger_list.append(Passenger(1, random.randint(2,10))) 
            if self.timer == 260:
                passenger_list.append(Passenger(4, 1))
                passenger_list.append(Passenger(5, 1)) 
                passenger_list.append(Passenger(9, 1)) 
                passenger_list.append(Passenger(10, 1))
                passenger_list.append(Passenger(4, 1)) 
                passenger_list.append(Passenger(4, 1)) 
                passenger_list.append(Passenger(7, 1))
            if self.timer == 280:
                passenger_list.append(Passenger(4, 1))
                passenger_list.append(Passenger(5, 1)) 
                passenger_list.append(Passenger(9, 1)) 
                passenger_list.append(Passenger(10, 1))
                passenger_list.append(Passenger(7, 1))
            # if self.timer == 300:
            #     passenger_list.append(Passenger(1, 4))
            #     passenger_list.append(Passenger(1, 5)) 
            #     passenger_list.append(Passenger(1, 9)) 
            #     passenger_list.append(Passenger(1, 10))
            #     passenger_list.append(Passenger(1, 4))
            #     passenger_list.append(Passenger(1, 4))
            #     passenger_list.append(Passenger(1, 7))
            # if self.timer == 320:
            #     passenger_list.append(Passenger(1, 4))
            #     passenger_list.append(Passenger(1, 5)) 
            #     passenger_list.append(Passenger(1, 9)) 
            #     passenger_list.append(Passenger(1, 10))
            #     passenger_list.append(Passenger(1, 7))
            # if self.timer == 420:
            #     passenger_list.append(Passenger(4, 1))
            #     passenger_list.append(Passenger(5, 1)) 
            #     passenger_list.append(Passenger(9, 1)) 
            #     passenger_list.append(Passenger(10, 1))
            #     passenger_list.append(Passenger(4, 1)) 
            #     passenger_list.append(Passenger(4, 1)) 
            #     passenger_list.append(Passenger(7, 1))
            # if self.timer == 430:
            #     passenger_list.append(Passenger(4, 1))
            #     passenger_list.append(Passenger(5, 1)) 
            #     passenger_list.append(Passenger(9, 1)) 
            #     passenger_list.append(Passenger(10, 1))
            #     passenger_list.append(Passenger(7, 1))
            # for i in range(noisy_passenger_num[0]):
            #     current_position = random.randint(1, self.max_floor)
            #     destination = random.randint(1,self.max_floor)
            #     while current_position == destination:
            #         destination = random.randint(1, self.max_floor)
            #     passenger = Passenger(current_position, destination)
            #     passenger_list.append(passenger)

        return passenger_list

generator = Passenger_Generator(0)

passenger_list = generator.passenger_list

for passenger in passenger_list:
    print("Current Position:", passenger.get_current_position())
    print("Destination:", passenger.get_destination())
    print("Waiting Time:", passenger.get_waiting_time())
    print()