import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from grid_world import GridWorld

class QLearningAgent:
    def __init__(self, 
                 env: GridWorld,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 exploration_rate: float = 0.3,
                 min_exploration: float = 0.01,
                 max_steps: int = 50):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration = min_exploration
        self.max_steps = max_steps
        
        # Initialize Q-table
        self.q_table = {}
        
        # Training history
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_successes = []
        
    def get_state_key(self, state: Tuple[int, int]) -> str:
        """Convert state to string key for Q-table"""
        return f"{state[0]}_{state[1]}"
    
    def get_q_value(self, state: Tuple[int, int], action: int) -> float:
        """Get Q-value for state-action pair"""
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(4)  # 4 actions
        return self.q_table[state_key][action]
    
    def choose_action(self, state: Tuple[int, int]) -> int:
        """Choose action using ε-greedy policy"""
        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, 4)  # Random action
        else:
            state_key = self.get_state_key(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(4)
            return np.argmax(self.q_table[state_key])
    
    def update_q_value(self, state: Tuple[int, int], action: int, reward: float,
                      next_state: Tuple[int, int]) -> None:
        """Update Q-value using Q-learning update rule"""
        current_q = self.get_q_value(state, action)
        next_state_key = self.get_state_key(next_state)
        
        # Get max Q-value for next state
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(4)
        max_next_q = np.max(self.q_table[next_state_key])
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        # Update Q-table
        state_key = self.get_state_key(state)
        self.q_table[state_key][action] = new_q
    
    def train(self, n_episodes: int = 1000) -> pd.DataFrame:
        """Train the agent for n_episodes"""
        training_data = []
        
        for episode in range(n_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            episode_success = False
            
            for step in range(self.max_steps):
                # Choose and take action
                action = self.choose_action(state)
                next_state, reward, done, success = self.env.step(action)
                
                # Update Q-value
                self.update_q_value(state, action, reward, next_state)
                
                # Update episode statistics
                total_reward += reward
                steps += 1
                episode_success = success
                
                # Store training data
                training_data.append({
                    'episode': episode,
                    'step': step,
                    'agent_x': state[0],
                    'agent_y': state[1],
                    'action': action,
                    'reward': reward,
                    'done': done,
                    'success': success,
                    'episode_success': episode_success
                })
                
                # Update state
                state = next_state
                
                if done:
                    break
            
            # Update exploration rate
            self.exploration_rate = max(
                self.min_exploration,
                self.exploration_rate * 0.995
            )
            
            # Store episode statistics
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            self.episode_successes.append(episode_success)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                success_rate = np.mean(self.episode_successes[-100:]) * 100
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode + 1}/{n_episodes}, "
                      f"Success Rate: {success_rate:.2f}%, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Exploration: {self.exploration_rate:.3f}")
        
        return pd.DataFrame(training_data)
    
    def save_q_table(self, filename: str) -> None:
        """Save Q-table to CSV file"""
        q_data = []
        for state_key, q_values in self.q_table.items():
            x, y = map(int, state_key.split('_'))
            for action, q_value in enumerate(q_values):
                q_data.append({
                    'state_x': x,
                    'state_y': y,
                    'action': action,
                    'q_value': q_value
                })
        pd.DataFrame(q_data).to_csv(filename, index=False)
    
    def load_q_table(self, filename: str) -> None:
        """Load Q-table from a file"""
        df = pd.read_csv(filename)
        self.q_table = {}
        for _, row in df.iterrows():
            state = (row['agent_x'], row['agent_y'])
            q_values = np.array([
                row['q_right'],
                row['q_down'],
                row['q_left'],
                row['q_up']
            ])
            self.q_table[state] = q_values 