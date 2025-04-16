import numpy as np
import pandas as pd
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from student_learning_env import StudentLearningEnvironment

class QLearningAgent:
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, 
                 exploration_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}
        self.reward_history = []
        self.state_value_history = []
        self.best_reward = float('-inf')
        self.stagnation_count = 0
        self.rolling_avg_reward = None
        self.rolling_window = 5
        # Add state normalization ranges
        self.state_ranges = {
            'study_hours': (0, 10),
            'sleep_hours': (0, 10),
            'stress_level': (0, 5),
            'health_status': (0, 1)
        }
        
    def normalize_state(self, state: Tuple) -> Tuple:
        """Normalize state values to [0, 1] range"""
        study, sleep, stress, health = state
        return (
            (study - self.state_ranges['study_hours'][0]) / 
            (self.state_ranges['study_hours'][1] - self.state_ranges['study_hours'][0]),
            (sleep - self.state_ranges['sleep_hours'][0]) / 
            (self.state_ranges['sleep_hours'][1] - self.state_ranges['sleep_hours'][0]),
            (stress - self.state_ranges['stress_level'][0]) / 
            (self.state_ranges['stress_level'][1] - self.state_ranges['stress_level'][0]),
            health  # Already binary
        )
    
    def get_state_key(self, state: Tuple) -> str:
        """Convert normalized state tuple to string key"""
        normalized_state = self.normalize_state(state)
        return f"{normalized_state[0]:.2f}_{normalized_state[1]:.2f}_{normalized_state[2]:.2f}_{normalized_state[3]}"
    
    def get_q_value(self, state: Tuple, action: int) -> float:
        """Get Q-value for state-action pair"""
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(5)  # 5 possible actions
        return self.q_table[state_key][action]
    
    def update_q_value(self, state: Tuple, action: int, reward: float, next_state: Tuple):
        """Update Q-value using Q-learning update rule"""
        current_q = self.get_q_value(state, action)
        next_state_key = self.get_state_key(next_state)
        
        # Get max Q-value for next state
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(5)
        max_next_q = np.max(self.q_table[next_state_key])
        
        # Q-learning update with adaptive learning rate
        state_key = self.get_state_key(state)
        learning_rate = self.learning_rate * (1 + np.exp(-len(self.q_table[state_key])))
        
        new_q = current_q + learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        # Update Q-table
        self.q_table[state_key][action] = new_q
        
    def choose_action(self, state: Tuple) -> int:
        """Choose action using ε-greedy policy"""
        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, 5)  # Random action
        else:
            state_key = self.get_state_key(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(5)
            return np.argmax(self.q_table[state_key])
    
    def calculate_reward(self, state: Tuple, action: int, next_state: Tuple, episode: int) -> float:
        """Calculate reward based on state transitions and episode progress"""
        # Calculate improvements in each dimension
        study_improvement = next_state[0] - state[0]
        sleep_improvement = next_state[1] - state[1]
        stress_improvement = state[2] - next_state[2]  # Lower stress is better
        health_improvement = next_state[3] - state[3]
        
        # Calculate penalties for extreme values
        study_penalty = max(0, abs(next_state[0] - 5) - 2)  # Fixed range
        sleep_penalty = max(0, abs(next_state[1] - 7) - 2)  # Fixed range
        stress_penalty = max(0, next_state[2] - 2)  # Fixed threshold
        
        # Base weights with action-specific adjustments
        base_weights = {
            'study': 0.4,
            'sleep': 0.3,
            'stress': 0.2,
            'health': 0.1
        }
        
        # Action-specific weight adjustments
        action_weights = {
            0: {'study': 13.0, 'sleep': 0.0, 'stress': 0.0, 'health': 0.0},  # Increase study
            1: {'study': 0.0, 'sleep': 13.0, 'stress': 0.0, 'health': 0.0},  # Increase sleep
            2: {'study': 0.0, 'sleep': 0.0, 'stress': 13.0, 'health': 0.0},  # Reduce stress
            3: {'study': 0.0, 'sleep': 0.0, 'stress': 0.0, 'health': 13.0},  # Improve health
            4: {'study': 0.1, 'sleep': 0.1, 'stress': 0.1, 'health': 0.1}   # Seek help
        }
        
        # Apply action-specific weights
        weights = {
            'study': base_weights['study'] + action_weights[action]['study'],
            'sleep': base_weights['sleep'] + action_weights[action]['sleep'],
            'stress': base_weights['stress'] + action_weights[action]['stress'],
            'health': base_weights['health'] + action_weights[action]['health']
        }
        
        improvement_reward = (
            study_improvement * weights['study'] +
            sleep_improvement * weights['sleep'] +
            stress_improvement * weights['stress'] +
            health_improvement * weights['health']
        )
        
        # Calculate bonus for good states with action-specific multipliers
        bonus = 0
        if 4 <= next_state[0] <= 6:
            bonus += 32.0 * (1 + 0.5 * (action == 0))  # Extra bonus for study action
        if 6 <= next_state[1] <= 8:
            bonus += 32.0 * (1 + 0.5 * (action == 1))  # Extra bonus for sleep action
        if next_state[2] <= 2:
            bonus += 32.0 * (1 + 0.5 * (action == 2))  # Extra bonus for stress action
        if next_state[3] == 1:
            bonus += 32.0 * (1 + 0.5 * (action == 3))  # Extra bonus for health action
        
        # Calculate penalties with action-specific reductions
        penalty_weights = {
            'study': 7.0 * (1 - 0.5 * (action == 0)),  # Reduced penalty for study action
            'sleep': 7.0 * (1 - 0.5 * (action == 1)),  # Reduced penalty for sleep action
            'stress': 6.5 * (1 - 0.5 * (action == 2))   # Reduced penalty for stress action
        }
        
        total_penalty = (
            study_penalty * penalty_weights['study'] +
            sleep_penalty * penalty_weights['sleep'] +
            stress_penalty * penalty_weights['stress']
        )
        
        # Final reward with action-specific scaling and state-based adjustments
        base_reward = improvement_reward + bonus - total_penalty
        state_factor = 1.0
        
        # Adjust reward based on current state and action appropriateness
        if state[0] < 4 and action == 0:  # Low study, increasing study
            state_factor *= 14.0
        elif state[0] > 6 and action == 0:  # High study, increasing study
            state_factor *= 0.00000000000001
            
        if state[1] < 6 and action == 1:  # Low sleep, increasing sleep
            state_factor *= 14.0
        elif state[1] > 8 and action == 1:  # High sleep, increasing sleep
            state_factor *= 0.00000000000001
            
        if state[2] > 2 and action == 2:  # High stress, reducing stress
            state_factor *= 14.0
        elif state[2] <= 2 and action == 2:  # Low stress, reducing stress
            state_factor *= 0.00000000000001
            
        if state[3] == 0 and action == 3:  # Poor health, improving health
            state_factor *= 14.0
        elif state[3] == 1 and action == 3:  # Good health, improving health
            state_factor *= 0.00000000000001
            
        # Scale down seek help for good states
        if action == 4 and all([
            state[0] >= 4,  # Good study
            state[1] >= 6,  # Good sleep
            state[2] <= 2,  # Low stress
            state[3] == 1   # Good health
        ]):
            state_factor *= 0.0000000000000001
        
        reward = base_reward * state_factor * (65.0 if action != 4 else 1.0)  # Increased scaling for non-seek help actions
        
        return reward
    
    def get_state_value(self, state: Tuple) -> float:
        """Calculate the value of a state"""
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            return 0
        return np.max(self.q_table[state_key])
    
    def train(self, data: pd.DataFrame, n_episodes: int = 100):
        """Train the agent on the dataset"""
        # Initialize Q-table with all possible states
        for _, row in data.iterrows():
            state = (
                row['study_hours'],
                row['sleep_hours'],
                row['stress_level'],
                row['health_status']
            )
            self.get_q_value(state, 0)
        
        for episode in range(n_episodes):
            total_reward = 0
            episode_state_values = []
            
            # Calculate rolling average reward
            if len(self.reward_history) >= self.rolling_window:
                self.rolling_avg_reward = np.mean(self.reward_history[-self.rolling_window:])
            
            # Dynamic exploration rate with episode-based adjustments
            if episode > 0:
                current_reward = self.reward_history[-1]
                if self.rolling_avg_reward is not None and current_reward < self.rolling_avg_reward:
                    self.stagnation_count += 1
                else:
                    self.stagnation_count = max(0, self.stagnation_count - 1)
                
                # More dynamic exploration adjustment
                base_exploration = self.exploration_rate * (1 - (episode/n_episodes)**1.000000000000001)  # Much slower decay
                if self.stagnation_count >= 2:  # Reduced threshold
                    current_exploration = min(1.0, base_exploration * (1 + self.stagnation_count / 1.0000000000001))  # Much faster increase
                else:
                    current_exploration = max(0.9999999999, base_exploration)  # Higher minimum
            else:
                current_exploration = self.exploration_rate
            
            # Shuffle the data for each episode
            shuffled_data = data.sample(frac=1).reset_index(drop=True)
            
            for _, row in shuffled_data.iterrows():
                state = (
                    row['study_hours'],
                    row['sleep_hours'],
                    row['stress_level'],
                    row['health_status']
                )
                
                action = self.choose_action(state) if np.random.random() < current_exploration else row['action']
                next_state = (
                    row['next_study_hours'],
                    row['next_sleep_hours'],
                    row['next_stress_level'],
                    row['next_health_status']
                )
                
                reward = self.calculate_reward(state, action, next_state, episode)
                self.update_q_value(state, action, reward, next_state)
                total_reward += reward
                episode_state_values.append(self.get_state_value(state))
            
            self.reward_history.append(total_reward)
            self.state_value_history.append(np.mean(episode_state_values))
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{n_episodes}, "
                      f"Total Reward: {total_reward:.2f}, "
                      f"Avg State Value: {np.mean(episode_state_values):.2f}, "
                      f"Exploration: {current_exploration:.2f}, "
                      f"Stagnation: {self.stagnation_count}")
    
    def plot_learning_curve(self):
        """Plot the learning curve"""
        plt.figure(figsize=(12, 6))
        
        # Plot rewards
        plt.subplot(1, 2, 1)
        plt.plot(self.reward_history)
        plt.title('Q-Learning Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        # Plot state values
        plt.subplot(1, 2, 2)
        plt.plot(self.state_value_history)
        plt.title('Average State Value')
        plt.xlabel('Episode')
        plt.ylabel('State Value')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('learning_curve.png')
        plt.close()
    
    def get_optimal_policy(self) -> Dict[str, int]:
        """Get the optimal policy from the learned Q-table"""
        policy = {}
        for state_key, q_values in self.q_table.items():
            policy[state_key] = np.argmax(q_values)
        return policy
    
    def print_optimal_actions(self, env):
        """Print optimal actions for different states"""
        print("\nOptimal Actions for Different States:")
        print("Format: (Study, Sleep, Stress, Health) -> Action")
        
        # Sample some states
        sample_states = [
            (2, 4, 3, 0),  # Poor conditions
            (6, 8, 1, 1),  # Good conditions
            (8, 6, 2, 1),  # High study, moderate sleep
            (4, 10, 0, 1)  # Low study, high sleep
        ]
        
        for state in sample_states:
            state_key = self.get_state_key(state)
            if state_key in self.q_table:
                action = np.argmax(self.q_table[state_key])
                print(f"{state} -> {env.actions[action]}")

# Main execution
if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv('student_learning_data.csv')
    
    # Create and train the agent
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.1
    )
    
    print("Starting Q-learning training...")
    agent.train(data, n_episodes=100)
    
    # Plot learning curve
    agent.plot_learning_curve()
    
    # Print optimal actions
    env = StudentLearningEnvironment()  # Import from student_learning_env.py
    agent.print_optimal_actions(env)
    
    print("\nTraining complete! Learning curve saved as 'learning_curve.png'") 