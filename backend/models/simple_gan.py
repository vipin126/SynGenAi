import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple

class Generator(nn.Module):
    """Simple Generator Network"""
    
    def __init__(self, latent_dim: int = 100, output_channels: int = 3, img_size: int = 64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        self.model = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 64 x 32 x 32
            nn.ConvTranspose2d(64, output_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # output_channels x 64 x 64
        )
    
    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    """Simple Discriminator Network"""
    
    def __init__(self, input_channels: int = 3, img_size: int = 64):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Input: input_channels x 64 x 64
            nn.Conv2d(input_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256 x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 512 x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.model(img).view(-1, 1)


class SimpleGAN:
    """GAN Trainer"""
    
    def __init__(self, latent_dim: int = 100, img_size: int = 64, channels: int = 3, lr: float = 0.0002):
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.generator = Generator(latent_dim, channels, img_size).to(self.device)
        self.discriminator = Discriminator(channels, img_size).to(self.device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        self.losses = {'g_loss': [], 'd_loss': []}
    
    def train(self, dataloader: DataLoader, epochs: int = 100, save_interval: int = 10):
        """Train the GAN"""
        
        for epoch in range(epochs):
            g_loss_epoch = 0
            d_loss_epoch = 0
            
            for i, real_imgs in enumerate(dataloader):
                batch_size = real_imgs.size(0)
                real_imgs = real_imgs.to(self.device)
                
                # Labels
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # ============ Train Discriminator ============
                self.d_optimizer.zero_grad()
                
                # Real images
                real_output = self.discriminator(real_imgs)
                d_loss_real = self.criterion(real_output, real_labels)
                
                # Fake images
                z = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
                fake_imgs = self.generator(z)
                fake_output = self.discriminator(fake_imgs.detach())
                d_loss_fake = self.criterion(fake_output, fake_labels)
                
                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()
                
                # ============ Train Generator ============
                self.g_optimizer.zero_grad()
                
                z = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
                fake_imgs = self.generator(z)
                fake_output = self.discriminator(fake_imgs)
                g_loss = self.criterion(fake_output, real_labels)
                
                g_loss.backward()
                self.g_optimizer.step()
                
                g_loss_epoch += g_loss.item()
                d_loss_epoch += d_loss.item()
            
            # Average losses
            g_loss_avg = g_loss_epoch / len(dataloader)
            d_loss_avg = d_loss_epoch / len(dataloader)
            
            self.losses['g_loss'].append(g_loss_avg)
            self.losses['d_loss'].append(d_loss_avg)
            
            print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss_avg:.4f} | G Loss: {g_loss_avg:.4f}")
    
    def generate(self, num_samples: int = 100) -> np.ndarray:
        """Generate synthetic samples"""
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, 1, 1).to(self.device)
            fake_imgs = self.generator(z)
            
            # Convert to numpy and denormalize
            fake_imgs = fake_imgs.cpu().numpy()
            fake_imgs = (fake_imgs + 1) * 127.5
            fake_imgs = fake_imgs.astype(np.uint8)
            
            # Change from (N, C, H, W) to (N, H, W, C)
            fake_imgs = np.transpose(fake_imgs, (0, 2, 3, 1))
        
        return fake_imgs
    
    def save_models(self, path: str):
        """Save generator and discriminator"""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
        }, path)
    
    def load_models(self, path: str):
        """Load saved models"""
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
