import os
import argparse
from utils.data import load_stock_data
from environment.trading_env import TradingEnvironment
from models.dqn import DQNAgent
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train DQN Trading Agent')
    parser.add_argument('--symbol', type=str, default='AAPL',
                        help='Stock symbol to trade')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                        help='Start date for training data')
    parser.add_argument('--end-date', type=str, default='2024-02-14',
                        help='End date for training data')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of episodes to train')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--initial-balance', type=float, default=10000,
                        help='Initial balance for trading')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save model checkpoints')
    return parser.parse_args()

def plot_results(total_rewards, moving_avg_rewards, filepath):
    """Plot and save training results."""
    plt.figure(figsize=(10, 5))
    plt.plot(total_rewards, label='Episode Reward')
    plt.plot(moving_avg_rewards, label='Moving Average (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig(filepath)
    plt.close()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    try:
        # Load and preprocess data
        print(f"Loading data for {args.symbol}...")
        data = load_stock_data(args.symbol, args.start_date, args.end_date)
        print(f"Successfully loaded {len(data)} data points")
        
        # Create environment and agent
        env = TradingEnvironment(data, initial_balance=args.initial_balance)
        state_size = 4  # Close, SMA5, SMA20, Returns
        action_size = 3  # Hold, Buy, Sell
        agent = DQNAgent(state_size, action_size)
        
        # Training loop
        total_rewards = []
        moving_avg_rewards = []
        best_reward = float('-inf')
        
        print("\nStarting training...")
        print(f"Training on {args.symbol} from {args.start_date} to {args.end_date}")
        print(f"Initial balance: ${args.initial_balance:.2f}")
        print("-" * 50)
        
        for episode in range(args.episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Choose and perform action
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                
                # Store experience in memory
                if next_state is not None:
                    agent.remember(state, action, reward, next_state, done)
                
                state = next_state if next_state is not None else state
                total_reward += reward
                
                # Train the agent
                agent.replay(args.batch_size)
                
                if done:
                    # Update target network every episode
                    agent.update_target_model()
                    
                    # Save best model
                    if total_reward > best_reward:
                        best_reward = total_reward
                        agent.save(os.path.join(args.model_dir, 'best_model.pth'))
                    
                    # Calculate moving average
                    total_rewards.append(total_reward)
                    moving_avg = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
                    moving_avg_rewards.append(moving_avg)
                    
                    print(f"Episode: {episode+1}/{args.episodes}")
                    print(f"Total Reward: {total_reward:.2f}")
                    print(f"Moving Average Reward: {moving_avg:.2f}")
                    print(f"Final Balance: ${env.balance:.2f}")
                    print(f"Epsilon: {agent.epsilon:.4f}")
                    print("-" * 50)
        
        # Save final model and plot results
        agent.save(os.path.join(args.model_dir, 'final_model.pth'))
        plot_results(total_rewards, moving_avg_rewards, os.path.join(args.model_dir, 'training_results.png'))
        
        print("\nTraining completed!")
        print(f"Best reward achieved: {best_reward:.2f}")
        print(f"Models saved in {args.model_dir}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Verify the stock symbol exists and is active")
        print("2. Check your internet connection")
        print("3. Ensure the date range is valid")
        print("4. Try with a different stock symbol or date range")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 