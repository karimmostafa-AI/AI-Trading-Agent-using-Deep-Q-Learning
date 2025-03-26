import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        """
        Initialize the Deep Q-Network.
        
        Args:
            state_size (int): Size of the input state
            action_size (int): Number of possible actions
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        """Forward pass through the network."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        """
        Initialize the DQN Agent.
        
        Args:
            state_size (int): Size of the input state
            action_size (int): Number of possible actions
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Neural Network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        """Choose action based on epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_size)
            
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state)
            return torch.argmax(q_values).item()
        
    def replay(self, batch_size):
        """Train the network using experience replay."""
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([i[0] for i in batch]).to(self.device)
        actions = torch.LongTensor([i[1] for i in batch]).to(self.device)
        rewards = torch.FloatTensor([i[2] for i in batch]).to(self.device)
        next_states = torch.FloatTensor([i[3] for i in batch]).to(self.device)
        dones = torch.FloatTensor([i[4] for i in batch]).to(self.device)
        
        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        # Compute loss and update weights
        loss = self.criterion(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_model(self):
        """Update target network weights with current network weights."""
        self.target_model.load_state_dict(self.model.state_dict())
        
    def save(self, filepath):
        """Save model weights to file."""
        torch.save(self.model.state_dict(), filepath)
        
    def load(self, filepath):
        """Load model weights from file."""
        self.model.load_state_dict(torch.load(filepath))
        self.target_model.load_state_dict(self.model.state_dict()) 