import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """
    Generator network for original MNIST images (28x28).
    Takes latent vectors from various latent generators and produces images.
    """
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        # Initial projection and reshape
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),  # Project to 7x7 feature map
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True)
        )
        
        # Upsampling layers: 7x7 -> 14x14 -> 28x28
        self.upsampling = nn.Sequential(
            # Block 1: 7x7 -> 14x14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Block 2: 14x14 -> 28x28
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
    def forward(self, z):
        # Project and reshape to initial feature map
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)  # Reshape to [batch, 128, 7, 7]
        
        # Apply upsampling blocks to reach 28x28
        img = self.upsampling(x)
        return img


class Discriminator(nn.Module):
    """
    Discriminator (Critic) network for original MNIST images (28x28).
    For WGAN-GP, the final output is a raw scalar (without a sigmoid).
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Downsampling layers: 28x28 -> 14x14 -> 7x7
        self.conv = nn.Sequential(
            # Block 1: 28x28 -> 14x14
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 2: 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            # Removed BatchNorm2d for WGAN-GP stability
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Final linear layer (no activation)
        self.fc = nn.Linear(128 * 7 * 7, 1)
        
    def forward(self, x):
        features = self.conv(x)
        features_flat = features.view(features.size(0), -1)  # Flatten
        validity = self.fc(features_flat)
        return validity
