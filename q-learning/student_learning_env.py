import numpy as np
import pandas as pd
import random
from typing import Tuple, Dict

class StudentLearningEnvironment:
    def __init__(self):
        # Define possible states
        self.study_hours = [2, 4, 6, 8]  # Hours of study
        self.sleep_hours = [4, 6, 8, 10]  # Hours of sleep
        self.stress_levels = [0, 1, 2, 3]  # Stress levels (0=low, 3=high)
        self.health_status = [0, 1]  # 0=poor, 1=good
        
        # Define possible actions
        self.actions = {
            0: "Increase study time",
            1: "Decrease study time",
            2: "Take a break",
            3: "Seek help",
            4: "Maintain current strategy"
        }
        
        # Define state space
        self.state_space = []
        for study in self.study_hours:
            for sleep in self.sleep_hours:
                for stress in self.stress_levels:
                    for health in self.health_status:
                        self.state_space.append((study, sleep, stress, health))
        
        # Initialize Q-table
        self.q_table = np.zeros((len(self.state_space), len(self.actions)))
        
        # Define rewards
        self.reward_rules = {
            'optimal_study': 10,  # 6-8 hours study
            'good_sleep': 5,      # 7-9 hours sleep
            'low_stress': 3,      # stress level 0-1
            'good_health': 2,     # health status 1
            'over_study': -5,     # >8 hours study
            'under_sleep': -3,    # <6 hours sleep
            'high_stress': -4,    # stress level 2-3
            'poor_health': -2     # health status 0
        }

    def get_state_index(self, state: Tuple) -> int:
        """Convert state tuple to index in state space"""
        return self.state_space.index(state)

    def calculate_reward(self, state: Tuple, action: int) -> float:
        """Calculate reward based on state and action"""
        study, sleep, stress, health = state
        reward = 0
        
        # Base rewards for current state
        if 6 <= study <= 8:
            reward += self.reward_rules['optimal_study']
        elif study > 8:
            reward += self.reward_rules['over_study']
            
        if 7 <= sleep <= 9:
            reward += self.reward_rules['good_sleep']
        elif sleep < 6:
            reward += self.reward_rules['under_sleep']
            
        if stress <= 1:
            reward += self.reward_rules['low_stress']
        else:
            reward += self.reward_rules['high_stress']
            
        if health == 1:
            reward += self.reward_rules['good_health']
        else:
            reward += self.reward_rules['poor_health']
        
        # Action-specific rewards
        if action == 0 and study < 8:  # Increase study time
            reward += 2
        elif action == 1 and study > 6:  # Decrease study time
            reward += 1
        elif action == 2 and stress > 1:  # Take a break
            reward += 3
        elif action == 3 and study < 6:  # Seek help
            reward += 4
        
        return reward

    def get_next_state(self, state: Tuple, action: int) -> Tuple:
        """Determine next state based on current state and action"""
        study, sleep, stress, health = state
        
        # Action effects
        if action == 0:  # Increase study time
            study = min(study + 2, 8)
            stress = min(stress + 1, 3)
        elif action == 1:  # Decrease study time
            study = max(study - 2, 2)
            stress = max(stress - 1, 0)
        elif action == 2:  # Take a break
            stress = max(stress - 1, 0)
            health = 1
        elif action == 3:  # Seek help
            study = min(study + 1, 8)
            stress = max(stress - 1, 0)
        
        # Natural state changes
        sleep = max(min(sleep + random.choice([-1, 0, 1]), 10), 4)
        stress = max(min(stress + random.choice([-1, 0, 1]), 3), 0)
        health = 1 if (sleep >= 7 and stress <= 1) else 0
        
        return (study, sleep, stress, health)

    def generate_dataset(self, n_samples: int = 100) -> pd.DataFrame:
        """Generate dataset of state-action-reward-next_state tuples"""
        data = []
        
        for _ in range(n_samples):
            # Random initial state
            state = (
                random.choice(self.study_hours),
                random.choice(self.sleep_hours),
                random.choice(self.stress_levels),
                random.choice(self.health_status)
            )
            
            # Random action
            action = random.choice(list(self.actions.keys()))
            
            # Calculate reward and next state
            reward = self.calculate_reward(state, action)
            next_state = self.get_next_state(state, action)
            
            data.append({
                'study_hours': state[0],
                'sleep_hours': state[1],
                'stress_level': state[2],
                'health_status': state[3],
                'action': action,
                'action_name': self.actions[action],
                'reward': reward,
                'next_study_hours': next_state[0],
                'next_sleep_hours': next_state[1],
                'next_stress_level': next_state[2],
                'next_health_status': next_state[3]
            })
        
        return pd.DataFrame(data)

# Generate and save the dataset
if __name__ == "__main__":
    env = StudentLearningEnvironment()
    df = env.generate_dataset(100)
    
    # Save to CSV
    df.to_csv('student_learning_data.csv', index=False)
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total samples: {len(df)}")
    print("\nState Statistics:")
    print(df[['study_hours', 'sleep_hours', 'stress_level', 'health_status']].describe())
    print("\nAction Distribution:")
    print(df['action_name'].value_counts())
    print("\nReward Statistics:")
    print(df['reward'].describe())
    
    # Print some example transitions
    print("\nExample Transitions:")
    for i in range(3):
        print(f"\nTransition {i+1}:")
        print(f"From: Study={df.iloc[i]['study_hours']}h, Sleep={df.iloc[i]['sleep_hours']}h, "
              f"Stress={df.iloc[i]['stress_level']}, Health={df.iloc[i]['health_status']}")
        print(f"Action: {df.iloc[i]['action_name']}")
        print(f"Reward: {df.iloc[i]['reward']}")
        print(f"To: Study={df.iloc[i]['next_study_hours']}h, Sleep={df.iloc[i]['next_sleep_hours']}h, "
              f"Stress={df.iloc[i]['next_stress_level']}, Health={df.iloc[i]['next_health_status']}") 