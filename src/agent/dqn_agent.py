# dqn_agent.py
# Deep Q-Network agent for RL
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class TwoHeadDQNAgent:
    def __init__(self, state_dim, jump_dim, content_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, buffer_size=10000, batch_size=64):
        self.state_dim = state_dim
        self.jump_dim = jump_dim
        self.content_dim = content_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.jump_q_net = QNetwork(state_dim, jump_dim).to(self.device)
        self.jump_target_net = QNetwork(state_dim, jump_dim).to(self.device)
        self.jump_target_net.load_state_dict(self.jump_q_net.state_dict())
        self.content_q_net = QNetwork(state_dim, content_dim).to(self.device)
        self.content_target_net = QNetwork(state_dim, content_dim).to(self.device)
        self.content_target_net.load_state_dict(self.content_q_net.state_dict())
        self.jump_optimizer = optim.Adam(self.jump_q_net.parameters(), lr=lr)
        self.content_optimizer = optim.Adam(self.content_q_net.parameters(), lr=lr)

    def update_target(self):
        self.jump_target_net.load_state_dict(self.jump_q_net.state_dict())
        self.content_target_net.load_state_dict(self.content_q_net.state_dict())

    def select_jump_action(self, state, mask=None):
        if np.random.rand() < self.epsilon:
            available = [i for i, m in enumerate(mask) if m] if mask is not None else list(range(self.jump_dim))
            if not available:
                raise ValueError("No available jump actions to select from (mask is empty)")
            return random.choice(available)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.jump_q_net(state)
        if mask is not None:
            q_values[0][[i for i, m in enumerate(mask) if not m]] = -float('inf')
        return torch.argmax(q_values).item()

    def select_content_action(self, state, mask=None):
        if np.random.rand() < self.epsilon:
            available = [i for i, m in enumerate(mask) if m] if mask is not None else list(range(self.content_dim))
            if not available:
                raise ValueError("No available content actions to select from (mask is empty)")
            return random.choice(available)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.content_q_net(state)
        if mask is not None:
            q_values[0][[i for i, m in enumerate(mask) if not m]] = -float('inf')
        return torch.argmax(q_values).item()

    def store(self, state, jump_action, content_action, reward, next_state, done):
        self.memory.append((state, jump_action, content_action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, jump_actions, content_actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        jump_actions = torch.LongTensor(jump_actions).unsqueeze(1).to(self.device)
        content_actions = torch.LongTensor(content_actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        # Update jump Q-network
        jump_q_values = self.jump_q_net(states).gather(1, jump_actions)
        with torch.no_grad():
            next_jump_q = self.jump_target_net(next_states).max(1)[0].unsqueeze(1)
            jump_target = rewards + self.gamma * next_jump_q * (1 - dones)
        jump_loss = nn.MSELoss()(jump_q_values, jump_target)
        self.jump_optimizer.zero_grad()
        jump_loss.backward()
        self.jump_optimizer.step()
        # Update content Q-network
        content_q_values = self.content_q_net(states).gather(1, content_actions)
        with torch.no_grad():
            next_content_q = self.content_target_net(next_states).max(1)[0].unsqueeze(1)
            content_target = rewards + self.gamma * next_content_q * (1 - dones)
        content_loss = nn.MSELoss()(content_q_values, content_target)
        self.content_optimizer.zero_grad()
        content_loss.backward()
        self.content_optimizer.step()
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class DQNAgent:
    def update_target(self):
        """Copy weights from q_net to target_net."""
        self.target_net.load_state_dict(self.q_net.state_dict())
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, buffer_size=10000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def select_action(self, state, mask=None):
        if np.random.rand() < self.epsilon:
            # Masked random action
            if mask is not None:
                available = [i for i, m in enumerate(mask) if m]
                if not available:
                    raise ValueError("No available actions to select from (mask is empty)")
                action = random.choice(available)
                if action >= self.action_dim:
                    raise ValueError(f"Selected action {action} is out of bounds for action_dim {self.action_dim}")
                return action
            action = random.randrange(self.action_dim)
            return action
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_net(state)
        if mask is not None:
            q_values[0][[i for i, m in enumerate(mask) if not m]] = -float('inf')
        action = torch.argmax(q_values).item()
        if action >= self.action_dim:
            raise ValueError(f"Selected action {action} is out of bounds for action_dim {self.action_dim}")
        return action

    def store(self, state, action, reward, next_state, done):
        if action >= self.action_dim:
            raise ValueError(f"Attempting to store out-of-bounds action {action} for action_dim {self.action_dim}")
        self.memory.append((state, action, reward, next_state, done))


    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards + self.gamma * next_q * (1 - dones)
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
