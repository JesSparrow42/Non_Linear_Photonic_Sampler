import torch
import torch.nn as nn
import torch.nn.functional as func

### TO-DO
# ...nothing?
###
# UNet structure for pix2pix
class Generator(nn.Module):
    """Modified U-Net w/o upstream"""
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim # Upstream handled by PTGenerator
        
        # Initial layer to project latent vector into spatial dimensions
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),  # Project latent vector to a 4x4 spatial map
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Downstreaming
        self.dec1 = self.conv_block(512, 256)   # Upsample to 8x8
        self.dec2 = self.conv_block(256, 128)   # Upsample to 16x16
        self.dec3 = self.conv_block(128, 64)    # Upsample to 32x32
        self.dec4 = self.conv_block(64, 32)     # Upsample to 64x64
        self.dec5 = self.conv_block(32, 16)     # Upsample to 128x128
        self.dec6 = self.conv_block(16, 8)      # Upsample to 256x256
        self.dec7 = nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1) # Output 512 x 512
        
    def conv_block(self, in_channels, out_channels):
        """Downstreaming block"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, z):
        # Start with latent vector input
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)  # Reshape to (batch_size, 512, 4, 4)
        
        # Decoder
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.dec6(x)
        x = self.dec7(x)
        
        return torch.tanh(x) # Val: -1 - 1


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 5 conv. layer as in paper
        self.model = nn.Sequential(
             nn.ZeroPad2d(2), # Control dimensions
             nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, bias=False),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(128,256, kernel_size=2, stride=2, padding=1, bias=False),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=1, bias=False),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(512, 1, kernel_size=2, stride=1, padding=0, bias=False),
             nn.Sigmoid()
         )

    def forward(self, x):
        x = x.to(torch.float32) # Ensure format
        return self.model(x).squeeze(3).squeeze(2).squeeze(1) # Remove singleton dimensions

