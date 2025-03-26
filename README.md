# AI Trading Agent using Deep Q-Learning

This project implements an AI trading agent using Deep Q-Learning (DQN) to make automated trading decisions based on historical stock market data.

## Features

- Deep Q-Learning implementation for trading decisions
- Real-time stock data fetching using yfinance
- Technical indicators (SMA, RSI, MACD)
- Customizable trading environment
- Training visualization and progress tracking
- Model checkpointing and loading

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── src/
│   ├── models/
│   │   └── dqn.py          # DQN model and agent implementation
│   ├── environment/
│   │   └── trading_env.py  # Trading environment
│   ├── utils/
│   │   └── data.py         # Data loading and preprocessing
│   └── train.py            # Main training script
├── requirements.txt
└── README.md
```

## Usage

1. Train the agent:
```bash
python src/train.py --symbol AAPL --start-date 2020-01-01 --end-date 2024-02-14 --episodes 500
```

Available arguments:
- `--symbol`: Stock symbol to trade (default: AAPL)
- `--start-date`: Start date for training data (default: 2020-01-01)
- `--end-date`: End date for training data (default: 2024-02-14)
- `--episodes`: Number of episodes to train (default: 500)
- `--batch-size`: Batch size for training (default: 32)
- `--initial-balance`: Initial balance for trading (default: 10000)
- `--model-dir`: Directory to save model checkpoints (default: models)

## Model Architecture

The DQN model consists of:
- Input layer: 4 features (Close price, SMA5, SMA20, Returns)
- Hidden layers: 2 fully connected layers with 64 units each
- Output layer: 3 units (Hold, Buy, Sell actions)

## Training Process

1. The agent interacts with the trading environment by:
   - Observing the current state (market data)
   - Choosing an action (Hold, Buy, or Sell)
   - Receiving a reward based on the profit/loss
   - Learning from the experience

2. The training uses:
   - Experience replay for stable learning
   - Target network for reducing overestimation
   - Epsilon-greedy exploration strategy

## Results

The training results are saved in the `models` directory:
- `best_model.pth`: Model weights with the highest reward
- `final_model.pth`: Model weights after training completion
- `training_results.png`: Plot of training progress

## License

This project is licensed under the MIT License. 