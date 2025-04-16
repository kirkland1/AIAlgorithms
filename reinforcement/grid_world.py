import numpy as np
import random
from typing import Tuple, List, Dict

class GridWorld:
    def __init__(self, size: int = 5):
        self.size = size
        self.grid = np.zeros((size, size))
        self.agent_pos = (0, 0)  # Start at top-left corner
        self.goal_pos = (size-1, size-1)  # Goal at bottom-right corner
        
        # Define obstacles (fixed positions for this example)
        self.obstacles = [
            (1, 1), (1, 3),
            (2, 2),
            (3, 1), (3, 3)
        ]
        
        # Define actions: up, right, down, left
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.action_names = {
            0: "Right",
            1: "Down",
            2: "Left",
            3: "Up"
        }
        self.reset()
    
    def reset(self) -> Tuple[int, int]:
        """Reset the environment to initial state"""
        self.agent_pos = (0, 0)
        return self.agent_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, bool]:
        """Take a step in the environment"""
        # Get action direction
        dx, dy = self.actions[action]
        
        # Calculate new position
        new_x = max(0, min(self.size-1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.size-1, self.agent_pos[1] + dy))
        new_pos = (new_x, new_y)
        
        # Check if new position is an obstacle
        if new_pos in self.obstacles:
            new_pos = self.agent_pos  # Stay in current position if hitting obstacle
        
        # Update agent position
        self.agent_pos = new_pos
        
        # Calculate reward
        reward = -1  # Default step penalty
        done = False
        success = False
        
        # Check if goal reached
        if self.agent_pos == self.goal_pos:
            reward = 20.0  # Goal reward
            done = True
            success = True
        
        return self.agent_pos, reward, done, success
    
    def get_state(self) -> Tuple[int, int]:
        """Get current state (agent position)"""
        return self.agent_pos
    
    def render(self) -> None:
        """Render the current state of the environment"""
        grid = np.zeros((self.size, self.size))
        grid[self.agent_pos] = 1  # Agent
        grid[self.goal_pos] = 2   # Goal
        print(grid) 