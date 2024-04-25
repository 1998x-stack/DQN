from torch import nn, optim
import torch, random

class DQNAgent:
    def __init__(self, model, buffer, config, env, target_model=None):
        self.model = model
        self.target_model = target_model
        print(f"Using {'Dueling ' if config.dueling else ''}DQN")
        print(model)
        
        self.env = env
        self.buffer = buffer
        
        self.DEVICE = config.DEVICE
        self.batch_size = config.batch_size
        self.discount_factor = config.discount_factor
        self.epsilon = config.epsilon
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.tau = config.tau
        self.l2_reg = config.l2_reg
        
        self.optimizer = optim.Adam(model.parameters(), lr=config.lr)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()  # Random action
        else:
            self.model.eval()
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.DEVICE)
                q_values = self.model(state)
                return q_values.max(1)[1].item()  # Best action based on current policy

    def train(self):
        if len(self.buffer) < self.batch_size:
            return  # Not enough samples to train

        self.model.train()
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.DEVICE)
        actions = actions.to(self.DEVICE)
        next_states = next_states.to(self.DEVICE)
        rewards = rewards.to(self.DEVICE)
        dones = dones.to(self.DEVICE)

        # Compute the current Q-values for the selected actions
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        if not self.target_model:
            # Compute the maximum Q-values for the next states
            next_q = self.model(next_states).max(1)[0]
        else:
            # Best actions according to the online model
            next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
            # Q-values from the target model for the next state
            next_q = self.target_model(next_states).gather(1, next_actions).squeeze(1)
        # Compute the target Q-values
        expected_q = rewards + self.discount_factor * next_q * (1 - dones)
        
        # Calculate L2 regularization loss
        if self.l2_reg:
            l2_loss = sum(param.pow(2).sum() for param in self.model.parameters()) 
            loss = nn.MSELoss()(current_q, expected_q.detach()) + self.l2_reg * l2_loss
        else:
            loss = nn.MSELoss()(current_q, expected_q.detach())
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        if self.target_model:
            self.soft_update(self.model, self.target_model, self.tau)
    
    def soft_update(self, online_model, target_model, tau):
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            # Update target model with a mix of its own weights and the online model's weights
            updated_weight = tau * online_param.data + (1 - tau) * target_param.data
            target_param.data.copy_(updated_weight)