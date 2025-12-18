import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple

class Generator(nn.Module):
    """Simple Generator for Tabular Data (MLP based)"""
    
    def __init__(self, latent_dim: int, output_dim: int):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            
            nn.Linear(512, output_dim),
            nn.Tanh() # Output in range [-1, 1]
        )
    
    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    """Simple Discriminator for Tabular Data (MLP based)"""
    
    def __init__(self, input_dim: int):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


class SimpleGAN:
    """GAN Trainer for Tabular Data"""
    
    def __init__(self, latent_dim: int = 10, img_size: int = 1, channels: int = 4, lr: float = 0.0002):
        # img_size is unused here, keeping signature for compatibility
        self.latent_dim = latent_dim
        self.output_dim = channels # For tabular, channels = num_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.generator = Generator(latent_dim, self.output_dim).to(self.device)
        self.discriminator = Discriminator(self.output_dim).to(self.device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
    def train(self, dataloader: DataLoader, epochs: int = 100):
        """Train the GAN (Not used directly in current main.py logic but good to have)"""
        pass
    
    def save_models(self, path: str):
        """Save generator and discriminator"""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
        }, path)
