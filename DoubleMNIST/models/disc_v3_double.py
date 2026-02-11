import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvResidualBlock(nn.Module):
    """
    Discriminative equivalent of ChannelCouplingLayer.
    Structure matches the Flow's subnet: 3x3 -> 1x1 -> 3x3.
    Difference: 
    - Input/Output channels are equal (no split).
    - Uses Residual Addition (x + net(x)) instead of Affine Coupling.
    """
    def __init__(self, channels, hidden_dim=64, dropout_p=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1), # 1x1 mixing
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            
            nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1)
        )
        # Zero-init last layer so block starts as identity
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x):
        return x + self.net(x)

class LinearResidualBlock(nn.Module):
    """
    Discriminative equivalent of SimpleFlow (Linear) layer.
    Structure matches your SimpleFlow: Linear -> Linear -> Linear.
    """
    def __init__(self, dim, hidden_dim=2048, dropout_p=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=dropout_p),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=dropout_p),
            
            nn.Linear(hidden_dim, dim)
        )
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x):
        return x + self.net(x)

class PseudoResNet(nn.Module):
    def __init__(self, num_classes=20, dropout_p=0.0): # num_classes=20 for 2 digits (10+10)
        super().__init__()
        
        # --- SCALE 1 ---
        # 1. Downsampling (Replaces Squeeze)
        # (1, 28, 56) -> (4, 14, 28)
        self.downsample1 = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)
        
        # 2. Body (Replaces ConvFlow layers)
        self.scale1_layers = nn.ModuleList()
        for _ in range(8):
            self.scale1_layers.append(nn.Sequential(
                # Replaces Invertible1x1Conv
                nn.Conv2d(4, 4, kernel_size=1), 
                # Replaces ChannelCouplingLayer
                ConvResidualBlock(channels=4, hidden_dim=64, dropout_p=dropout_p)
            ))

        # --- SCALE 2 ---
        # 1. Downsampling (Replaces Squeeze2)
        # (4, 14, 28) -> (16, 7, 14)
        self.downsample2 = nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1)
        
        # 2. Body (Replaces ConvFlow layers)
        self.scale2_layers = nn.ModuleList()
        for _ in range(8):
            self.scale2_layers.append(nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=1),
                ConvResidualBlock(channels=16, hidden_dim=128, dropout_p=dropout_p)
            ))

        # --- GLOBAL TAIL ---
        flat_dim = 16 * 7 * 14 
        
        # Linear Body (Replaces SimpleFlow)
        self.linear_layers = nn.ModuleList()
        for _ in range(4):
            self.linear_layers.append(
                LinearResidualBlock(dim=flat_dim, hidden_dim=2048, dropout_p=dropout_p)
            )
            
        # Two separate heads for the two digits
        self.head_digit1 = nn.Linear(flat_dim, num_classes)
        self.head_digit2 = nn.Linear(flat_dim, num_classes)

    def forward(self, x):
        if x.dim() == 2: 
            x = x.view(-1, 1, 28, 56) # Handle DoubleMNIST shape
        
        x = self.downsample1(x) # 28x56 -> 14x28
        x = F.relu(x) 
        
        for layer in self.scale1_layers:
            x = layer(x)
            
        x = self.downsample2(x) # 14x28 -> 7x14
        x = F.relu(x)
        
        for layer in self.scale2_layers:
            x = layer(x)
            
        x = x.view(x.size(0), -1) # Flattens to (N, 1568)
        
        for layer in self.linear_layers:
            x = layer(x)
            
        # Classification
        logits1 = self.head_digit1(x) # (N, 10)
        logits2 = self.head_digit2(x) # (N, 10)
        
        return logits1, logits2