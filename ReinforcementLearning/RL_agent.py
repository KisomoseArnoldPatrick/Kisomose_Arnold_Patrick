# Assignment onetrain RL agent to navigate to cross the road with action right, left, right

#Import necessary librarires
import numpy as np
import random

# Environment setup
road_length = 5 # Theroad will have five positions 0 to 4
actions = ['left', 'right'] # Agent can move left or right

# Q- table Initialization. Initially, all Q-values are 0, as the agent has no prior knowledge.
Q = np.zeros((road_length, len(actions))) # A Q-table is created as a 5x2 matrix (5 states × 2 actions: left, right).

#Hyperparameters
episodes = 1000 # Number of iterations to train the agent (leads to better learning rate)
learning_rate = 0.8 #  how much the Q-value updates with new information
gamma = 0.9 # Discount factor for future rewards (how much the agent values future rewards)
epsilon = 0.3 # Exploration rate (helps the agent explore the environment)

# Training loop
for episode in range(episodes):
    state = 0 # Start position 0
    
    while state != 4: # Goal is to reach position 4
        # Epsilon-greedy action selection
        '''The agent balances exploration (trying new actions) and exploitation (using known good actions) using an epsilon-greedy strategy:

        With probability $ \epsilon $, choose a random action (explore).
        With probability $ 1 - \epsilon $, choose the action with the highest Q-value (exploit).'''
        if random.uniform(0,1) < epsilon: # If a random number (0 to 1) is less than $ \epsilon $ (0.3), it picks a random action (0 for left, 1 for right).
            action = random.randint(0,1) # Explore my random action
        else: # Otherwise, it picks the action with the highest Q-value for the current state
            action = np.argmax(Q[state]) # Exploit learned action
        
        # Taking an Action and Updating State
        if action == 0:  # Move left
            new_state = max(0, state - 1)
        else:  # Move right
            new_state = min(4, state + 1)

        # Reward
        reward = 1 if new_state == 4 else 0  # +1 if reached goal, else 0
        
        # Q-value Update
        Q[state, action] = Q[state, action] + learning_rate * (
            reward + gamma * np.max(Q[new_state]) - Q[state, action]
        )
        
        # Move to new state
        state = new_state
        
# Display learned Q-table
print("Learned Q-table:")
print(Q)

# Test the trained agent
state = 0
steps = 0
path = []
print("\nAgent's path to cross the road:")
while state != 4:
    action = np.argmax(Q[state])  # Choose best action
    if action == 0:
        state = max(0, state - 1)
        path.append("left")
    else:
        state = min(4, state + 1)
        path.append("right")
    steps += 1
    print(f"Step {steps}: Move {actions[action]} → Position {state}")

print(f"\nFinal path: {' → '.join(path)}")
print(f"Goal reached in {steps} steps!")
