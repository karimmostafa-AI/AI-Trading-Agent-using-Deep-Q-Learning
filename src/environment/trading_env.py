import numpy as np

ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL"}

def get_state(data, index):
    """Extract the current state from the data at given index."""
    return np.array([
        float(data.loc[index, 'Close']),
        float(data.loc[index, 'SMA_5']),
        float(data.loc[index, 'SMA_20']),
        float(data.loc[index, 'Returns'])
    ])

class TradingEnvironment:
    def __init__(self, data, initial_balance=10000):
        """
        Initialize the trading environment.
        
        Args:
            data (pd.DataFrame): Historical price data
            initial_balance (float): Starting balance for trading
        """
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state."""
        self.balance = self.initial_balance
        self.holdings = 0
        self.index = 0
        return get_state(self.data, self.index)
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (int): The action to take (0: HOLD, 1: BUY, 2: SELL)
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        current_price = float(self.data.loc[self.index, 'Close'])
        reward = 0
        info = {}
        
        # Execute trading action
        if action == 1 and self.balance >= current_price:  # BUY
            shares_to_buy = self.balance // current_price
            cost = shares_to_buy * current_price
            self.holdings += shares_to_buy
            self.balance -= cost
            info['action'] = 'BUY'
            info['shares'] = shares_to_buy
            info['price'] = current_price
            
        elif action == 2 and self.holdings > 0:  # SELL
            revenue = self.holdings * current_price
            self.balance += revenue
            info['action'] = 'SELL'
            info['shares'] = self.holdings
            info['price'] = current_price
            self.holdings = 0
            
        else:  # HOLD
            info['action'] = 'HOLD'
            
        # Move to next timestep
        self.index += 1
        done = self.index >= len(self.data) - 1
        
        # Calculate reward
        if done:
            # Final reward is the total profit/loss
            total_value = self.balance + (self.holdings * current_price)
            reward = total_value - self.initial_balance
        else:
            # Intermediate reward based on unrealized gains/losses
            total_value = self.balance + (self.holdings * current_price)
            reward = total_value - self.initial_balance
            
        next_state = get_state(self.data, self.index) if not done else None
        
        info['balance'] = self.balance
        info['holdings'] = self.holdings
        info['total_value'] = total_value
        
        return next_state, reward, done, info
    
    def render(self):
        """Display the current state of the environment."""
        price = float(self.data.loc[self.index, 'Close'])
        total_value = self.balance + (self.holdings * price)
        print(f"Day: {self.index}")
        print(f"Price: ${price:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Holdings: {self.holdings} shares")
        print(f"Total Value: ${total_value:.2f}")
        print("-" * 50) 