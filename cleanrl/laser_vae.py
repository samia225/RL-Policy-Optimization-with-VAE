import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import gymnasium as gym
from torch.utils.data import Dataset, DataLoader

# Conditional VAE for states and actions
class CVAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=1, hidden_dim=128):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def encode(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, state):
        x = torch.cat([z, state], dim=-1)
        return self.decoder(x)
    
    def forward(self, state, action):
        mu, log_var = self.encode(state, action)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, state), mu, log_var

# Dataset for collected optimal policy data
class OptimalPolicyDataset(Dataset):
    def __init__(self, states, actions):
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

# LASER Environment Wrapper
class LASERWrapper(gym.Wrapper):
    def __init__(self, env, vae, device='cuda'):
        super().__init__(env)
        self.vae = vae
        self.device = device
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
    def step(self, action):
        # Convert 1D LASER action to original action space
        state_tensor = torch.FloatTensor(self.state).unsqueeze(0).to(self.device)
        z = torch.FloatTensor([[action]]).to(self.device)  # Single latent dimension
        with torch.no_grad():
            original_action = self.vae.decode(z, state_tensor).cpu().numpy()[0]
        
        return self.env.step(original_action)

def collect_optimal_policy_data(agent, env, num_episodes=100):
    states, actions = [], []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                action, _, _, _ = agent.get_action_and_value(state_tensor)
                action = action.cpu().numpy()[0]
            
            states.append(state)
            actions.append(action)
            
            next_state, _, termination, truncation, _ = env.step(action)
            done = termination or truncation
            state = next_state
            
    return np.array(states), np.array(actions)

def train_cvae(states, actions, device='cuda', epochs=100, batch_size=128):
    # Initialize CVAE
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    cvae = CVAE(state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-3)
    
    # Create dataset and dataloader
    dataset = OptimalPolicyDataset(states, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for batch_states, batch_actions in dataloader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            
            # Forward pass
            recon_actions, mu, log_var = cvae(batch_states, batch_actions)
            
            # Reconstruction loss
            recon_loss = nn.MSELoss()(recon_actions, batch_actions)
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Total loss
            loss = recon_loss + 0.1 * kl_loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
    
    return cvae

def setup_laser_environment(env, agent, device='cuda'):
    # 1. Collect data using optimal policy
    states, actions = collect_optimal_policy_data(agent, env)
    
    # 2. Train CVAE on collected data
    cvae = train_cvae(states, actions, device=device)
    
    # 3. Create LASER wrapped environment
    laser_env = LASERWrapper(env, cvae, device=device)
    
    return laser_env, cvae