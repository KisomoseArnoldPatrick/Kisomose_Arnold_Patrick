# What is reinforcement learning?
'''Reinforcement learning (RL) is a type of machine learning where an agent learns to make decisions by taking actions in
an environment to maximize cumulative reward. The agent interacts with the environment, receives
feedback in the form of rewards or penalties, and adjusts its actions accordingly.'''

# Characteristics:
# - Trial and error: The agent learns from the consequences of its actions, exploring different strategies.
# - Delayed rewards: The agent may not receive immediate feedback, making it necessary to evaluate long-term outcomes.
# - Exploration vs. exploitation: The agent must balance exploring new actions and exploiting known rewarding actions.
# - Markov Decision Process (MDP): Many RL problems can be modeled as MDPs, where states, actions, and rewards are defined.
# - Policy and value functions: The agent learns a policy (mapping from states to actions) and value functions (estimating the expected reward).

# Common algorithms:
# - Q-Learning: A model-free algorithm that learns the value of actions in states.
# - Deep Q-Networks (DQN): Combines Q-learning with deep neural networks to handle high-dimensional state spaces.
# - Policy Gradient Methods: Directly optimize the policy by adjusting the parameters based on the gradient of expected rewards.
# - Actor-Critic Methods: Use both a policy (actor) and a value function (critic) to improve learning efficiency.

import numpy as np
import random

class RoadCrossingEnvironment:
    def __init__(self, road_width=5, car_probability=0.3):
        self.road_width = road_width  # Width of the road (0 to road_width-1)
        self.car_probability = car_probability  # Probability of car at each position
        self.reset()
    
    def reset(self):
        """Reset agent to starting position"""
        self.agent_position = 0  # Start at leftmost position
        self.cars = self._generate_cars()
        return self.agent_position
    
    def _generate_cars(self):
        """Generate random car positions for this episode"""
        cars = []
        for pos in range(1, self.road_width - 1):  # Cars can be in middle positions
            if random.random() < self.car_probability:
                cars.append(pos)
        return cars
    
    def step(self, action):
        """Take action and return next_state, reward, done"""
        # Actions: 0 = left, 1 = right
        if action == 0:  # Move left
            next_position = max(0, self.agent_position - 1)
        else:  # Move right
            next_position = min(self.road_width - 1, self.agent_position + 1)
        
        # Calculate reward
        reward = self._get_reward(next_position)
        
        # Check if episode is done
        done = (next_position == self.road_width - 1) or (next_position in self.cars)
        
        self.agent_position = next_position
        return next_position, reward, done
    
    def _get_reward(self, position):
        """Calculate reward based on position"""
        if position in self.cars:
            return -10  # Hit by car - large negative reward
        elif position == self.road_width - 1:
            return 20   # Successfully crossed - larger positive reward
        elif position > self.agent_position:  # Moved right (progress)
            return 1    # Small positive reward for progress
        else:
            return -0.5  # Penalty for not making progress
    
    def render(self):
        """Visualize current state"""
        road = ['-'] * self.road_width
        road[self.agent_position] = 'A'  # Agent
        for car_pos in self.cars:
            if car_pos != self.agent_position:
                road[car_pos] = 'C'  # Car
            else:
                road[car_pos] = 'X'  # Collision
        return '|' + ''.join(road) + '|'

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.15, discount_factor=0.95, epsilon=0.4):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.9995  # Much slower decay
        self.epsilon_min = 0.1       # Keep exploring
        
        # Initialize Q-table with slight bias toward moving right
        self.q_table = np.zeros((n_states, n_actions))
        # Give slight preference to moving right initially
        for state in range(n_states - 1):
            self.q_table[state, 1] = 0.1  # Small positive bias for "right"
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit
    
    def update_q_table(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning formula"""
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_agent():
    """Train the RL agent"""
    # Environment setup - make it easier initially
    env = RoadCrossingEnvironment(road_width=5, car_probability=0.15)  # Even lower car probability
    agent = QLearningAgent(n_states=5, n_actions=2)
    
    # Training parameters
    episodes = 3000  # More episodes
    rewards_per_episode = []
    success_rate_window = []
    
    print("Training agent to cross the road safely...")
    print("Road layout: |Start-...-...-...-Goal|")
    print("A = Agent, C = Car, X = Collision")
    print(f"Car probability: {env.car_probability:.1%}")
    print(f"Training for {episodes} episodes...")
    print("Key improvements:")
    print("- Q-table initialized with right-movement bias")
    print("- Slower epsilon decay to maintain exploration")
    print("- Reward system encourages forward progress\n")
    
    successful_episodes = 0
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        episode_successful = False
        
        while True:
            # Choose action
            action = agent.choose_action(state)
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Update Q-table
            agent.update_q_table(state, action, reward, next_state, done)
            
            total_reward += reward
            steps += 1
            
            if done:
                if next_state == env.road_width - 1:  # Successfully crossed
                    episode_successful = True
                    successful_episodes += 1
                break
            
            state = next_state
            
            # Prevent infinite loops
            if steps > 50:  # Reduced to prevent getting stuck
                break
        
        rewards_per_episode.append(total_reward)
        success_rate_window.append(1 if episode_successful else 0)
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Print progress every 300 episodes with more detail
        if (episode + 1) % 300 == 0:
            recent_success_rate = np.mean(success_rate_window[-300:]) * 100
            avg_reward = np.mean(rewards_per_episode[-300:])
            print(f"Episode {episode + 1}: Success Rate = {recent_success_rate:.1f}%, "
                  f"Avg Reward = {avg_reward:.2f}, Epsilon = {agent.epsilon:.3f}")
            print(f"Total Successful Episodes: {successful_episodes}")
            
            # Show current Q-table preferences
            print("Current Q-table preferences:")
            for s in range(agent.n_states):
                left_val = agent.q_table[s, 0]
                right_val = agent.q_table[s, 1]
                preference = "Right" if right_val > left_val else "Left"
                print(f"  State {s}: {preference} (L:{left_val:.2f}, R:{right_val:.2f})")
            print()
    
    print(f"Training completed!")
    print(f"Total successful episodes: {successful_episodes}/{episodes} ({successful_episodes/episodes*100:.1f}%)")
    
    return agent, env, rewards_per_episode, success_rate_window

def test_agent(agent, env, num_tests=10):
    """Test the trained agent"""
    print(f"\nTesting trained agent over {num_tests} episodes:")
    print("=" * 50)
    
    successes = 0
    
    for test in range(num_tests):
        state = env.reset()
        steps = 0
        path = [env.render()]
        
        # Use greedy policy (no exploration)
        original_epsilon = agent.epsilon
        agent.epsilon = 0
        
        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            path.append(env.render())
            steps += 1
            
            if done:
                if next_state == env.road_width - 1:
                    successes += 1
                    result = "SUCCESS!"
                else:
                    result = "COLLISION!"
                break
            
            state = next_state
            
            if steps > 20:  # Prevent infinite loops
                result = "TIMEOUT"
                break
        
        # Restore original epsilon
        agent.epsilon = original_epsilon
        
        print(f"\nTest {test + 1}: {result} (Steps: {steps})")
        for i, frame in enumerate(path):
            print(f"Step {i}: {frame}")
    
    success_rate = (successes / num_tests) * 100
    print(f"\nOverall Success Rate: {success_rate:.1f}% ({successes}/{num_tests})")
    
    return success_rate

def print_training_summary(rewards_per_episode, success_rate_window):
    """Print training summary statistics"""
    window_size = 50
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    if len(rewards_per_episode) >= window_size:
        recent_avg_reward = np.mean(rewards_per_episode[-window_size:])
        overall_avg_reward = np.mean(rewards_per_episode)
        print(f"Average Reward (last {window_size} episodes): {recent_avg_reward:.2f}")
        print(f"Average Reward (overall): {overall_avg_reward:.2f}")
    
    if len(success_rate_window) >= window_size:
        recent_success_rate = np.mean(success_rate_window[-window_size:]) * 100
        overall_success_rate = np.mean(success_rate_window) * 100
        print(f"Success Rate (last {window_size} episodes): {recent_success_rate:.1f}%")
        print(f"Success Rate (overall): {overall_success_rate:.1f}%")
    
    print("="*60)

def print_q_table(agent):
    """Print the learned Q-table with analysis"""
    print("\nLearned Q-table:")
    print("States: 0=Start, 1-3=Middle, 4=Goal")
    print("Actions: 0=Left, 1=Right")
    print("-" * 35)
    print("State | Left  | Right | Best Action")
    print("-" * 35)
    for state in range(agent.n_states):
        left_val = agent.q_table[state, 0]
        right_val = agent.q_table[state, 1]
        best_action = "Right" if right_val > left_val else "Left"
        if abs(right_val - left_val) < 0.01:
            best_action = "Similar"
        print(f"  {state}   | {left_val:5.2f} | {right_val:5.2f} | {best_action}")
    print("-" * 35)
    
    # Analysis
    print("\nQ-table Analysis:")
    total_updates = np.sum(np.abs(agent.q_table))
    if total_updates < 0.1:
        print("⚠️  Q-table has very small values - agent may not have learned much")
    
    # Check if agent prefers right movement (which should be the case)
    right_preference = 0
    for state in range(agent.n_states - 1):  # Exclude goal state
        if agent.q_table[state, 1] > agent.q_table[state, 0]:
            right_preference += 1
    
    print(f"States preferring 'Right': {right_preference}/{agent.n_states-1}")
    if right_preference < agent.n_states - 2:
        print("⚠️  Agent doesn't consistently prefer moving right")

if __name__ == "__main__":
    # Train the agent
    trained_agent, environment, rewards, success_rates = train_agent()
    
    # Print Q-table
    print_q_table(trained_agent)
    
    # Print training summary
    print_training_summary(rewards, success_rates)
    
    # Test the agent
    final_success_rate = test_agent(trained_agent, environment, num_tests=20)
    
    print(f"\nTraining completed! Final success rate: {final_success_rate:.1f}%")