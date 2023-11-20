# Elevator Simulation (Elevator_sim):

The env.py is composed of the gym environment construction and class of the agent, person.py is composed of the class "Person_Genarator" and class "Person", the render.py defines the class "Render" which is used for the visualision. test.py is for you to test our code. You can find the visualision result inside the "animation_buffer" folder.

instruction on test the environment:

1. Open the terminal in this folder: "Elevator_sim".

2. Create the conda environment in terminal by:
	conda env create -f environment.yaml
	
3. Activate conda environment in terminal by:
	conda activate Elevator_sim
	
4. Run the gym environment in terminal to test:
	python3 test.py
	
    You can change the settings of the elevator number(3 to 8) and the number of floors(8 to 30) in test.py, the gif animation is safed in the animation_buffer folder autonomously.
    
######################################################################
		      	  Neural Netwark
######################################################################
instruction on neural networks:

    We provided three neural netwarks in construction, the DQN is designed in train_dqn.py, LSTM is designed in train_lstm.py, the mixed input neural network is designed in train_nn.py
    
5. You can test our DQN by running:
	python3 train_dqn.py

6. You can test our LSTM by running:
	python3 train_lstm.py
	
######################################################################
		      	  Learning Agent
######################################################################
instruction on learning agent:
    
7. You can use the SARSA(0) to train the agent by running:
	python3 train_SARSA_0.py

8. You can use the SARSA(Lambda) to train the agent by running:
	python3 train_SARSA_lambda.py
The training result of these two network can be found in the elevator_qnetwork.pth.

######################################################################
		      	  Learning Agent
######################################################################
Use the learning result to control the elevator:

10. You can use the learning result to control the elevator by:
	python3 utility.py
