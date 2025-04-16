import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from grid_world import GridWorld
from q_learning_agent import QLearningAgent

def plot_grid_world(env: GridWorld, path: list = None) -> None:
    """Plot the grid world and agent's path"""
    plt.figure(figsize=(10, 8))  # Made figure wider to accommodate legend
    
    # Create grid
    grid = np.zeros((env.size, env.size))
    
    # Plot obstacles
    for obstacle in env.obstacles:
        grid[obstacle[0], obstacle[1]] = 1
    
    # Plot grid with origin at top-left
    plt.imshow(grid, cmap='binary', origin='upper')
    
    # Plot goal (using correct coordinates)
    plt.plot(env.goal_pos[1], env.goal_pos[0], 'g*', markersize=15)
    
    # Plot path if provided
    if path:
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        plt.plot(path_x, path_y, 'b-', linewidth=2)
        plt.plot(path_x[0], path_y[0], 'ro', markersize=10)
        plt.plot(path_x[-1], path_y[-1], 'r*', markersize=10)
    
    # Add grid lines and labels
    plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(0, env.size, 1))
    plt.yticks(np.arange(0, env.size, 1))
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Add coordinate labels in each cell
    for i in range(env.size):
        for j in range(env.size):
            plt.text(j, i, f'({j},{i})', ha='center', va='center', 
                    color='red' if (i,j) in env.obstacles else 'black',
                    fontsize=8)
    
    # Create custom legend outside the plot
    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='g', label='Goal', markersize=10),
        plt.Line2D([0], [0], color='b', label='Path', linewidth=2),
        plt.Line2D([0], [0], marker='o', color='r', label='Start', markersize=8),
        plt.Line2D([0], [0], marker='*', color='r', label='End', markersize=8)
    ]
    
    # Add legend outside the plot area
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5))
    
    plt.title('Grid World Environment')
    plt.tight_layout()  # Adjust layout to prevent legend cutoff
    plt.savefig('grid_world.png', bbox_inches='tight')  # Save with extra space for legend
    plt.close()

def plot_learning_curve(df: pd.DataFrame) -> None:
    """Plot the learning curve from training data"""
    plt.figure(figsize=(12, 6))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    rewards = df.groupby('episode')['reward'].sum()
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Plot success rate
    plt.subplot(1, 2, 2)
    success_rate = df.groupby('episode')['episode_success'].mean() * 100
    plt.plot(success_rate)
    plt.title('Success Rate')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('learning_curve.png')
    plt.close()

def main():
    # Create environment
    env = GridWorld(size=5)
    
    # Create agent
    agent = QLearningAgent(
        env=env,
        learning_rate=0.2,
        discount_factor=0.95,
        exploration_rate=0.3,
        min_exploration=0.05,
        max_steps=30
    )
    
    # Plot initial grid world
    plot_grid_world(env)
    print("Initial grid world saved as 'grid_world.png'")
    
    # Train agent
    print("Starting training...")
    df = agent.train(n_episodes=1000)
    
    # Save training data
    df.to_csv('training_data.csv', index=False)
    print(f"Training data saved with {len(df)} entries")
    
    # Save Q-table
    agent.save_q_table('q_table.csv')
    print("Q-table saved")
    
    # Plot learning curve
    plot_learning_curve(df)
    print("Learning curve plot saved as 'learning_curve.png'")
    
    # Print final statistics
    final_success_rate = df['episode_success'].mean() * 100
    avg_reward = df.groupby('episode')['reward'].sum().mean()
    
    print("\nFinal Training Statistics:")
    print(f"Total episodes: {df['episode'].nunique()}")
    print(f"Average reward per episode: {avg_reward:.2f}")
    print(f"Success rate: {final_success_rate:.2f}%")
    
    # Plot example path
    example_episode = df[df['episode'] == 999]  # Last episode
    path = [(row['agent_x'], row['agent_y']) for _, row in example_episode.iterrows()]
    plot_grid_world(env, path)
    print("Example path saved as 'grid_world.png'")
    
    # Print example entries
    print("\nExample entries from the dataset:")
    print(df.tail(10))

if __name__ == "__main__":
    main() 