import torch
import torch.nn as nn

class SimpleConvNet(nn.Module):
    """
    A lightweight Convolutional Network to predict s and t.
    It preserves spatial dimensions (padding=1 for 3x3 kernels).
    """
    def __init__(self, hidden_dim=64, dropout_p=0.0):
        super().__init__()
        self.net = nn.Sequential(
            # Input is 1 channel (grayscale), output hidden_dim
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1), # 1x1 conv mixing
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            # Output is 2 channels: one for s (scale), one for t (translation)
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1)
        )
        
        # Initialize the last layer to zeros is a common trick in Flows
        # so the flow starts as an identity function
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x):
        return self.net(x)

class ConvFlow(nn.Module):
    def __init__(self, num_layers=8, hidden_dim=64, dropout_p=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.masks = []
        
        # 1. Create Checkerboard Masks
        # We define a generic 28x28 mask
        mask = torch.zeros(1, 1, 28, 28)
        # Create a checkerboard pattern
        for i in range(28):
            for j in range(28):
                if (i + j) % 2 == 0:
                    mask[0, 0, i, j] = 1
        
        mask_even = mask
        mask_odd = 1 - mask
        
        for i in range(num_layers):
            # Alternating masks
            self.masks.append(mask_even if i % 2 == 0 else mask_odd)
            # The network that transforms the data
            self.layers.append(SimpleConvNet(hidden_dim, dropout_p=dropout_p))

    def forward(self, x):
        log_det_jac = 0
        # x is expected to be (N, 1, 28, 28)
        z = x 
        
        for i, layer in enumerate(self.layers):
            mask = self.masks[i].to(z.device)
            
            # Mask the input to the conditioner network
            masked_z = z * mask
            
            # Run the CNN
            out = layer(masked_z)
            s, t = out.chunk(2, dim=1) # Split channel-wise
            
            # Apply the affine coupling constraints
            # s and t must be 0 where the mask is 1 (the identity part)
            s = torch.tanh(s) * (1 - mask)
            t = t * (1 - mask)
            
            # Affine transformation
            z = z * torch.exp(s) + t
            
            # Log determinant is sum over all spatial dims
            log_det_jac += s.sum(dim=[1, 2, 3])
        
        z = z.view(z.size(0), -1)
        return z, log_det_jac

    def inverse(self, z):
        # z is expected to be (N, 1, 28, 28)
        x = z.view(-1, 1, 28, 28)
        
        for i, layer in reversed(list(enumerate(self.layers))):
            mask = self.masks[i].to(x.device)
            masked_x = x * mask
            
            out = layer(masked_x)
            s, t = out.chunk(2, dim=1)
            
            s = torch.tanh(s) * (1 - mask)
            t = t * (1 - mask)
            
            x = (x - t) * torch.exp(-s)
            
        return x
    

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFlow(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=1024, num_layers=8, dropout_p=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.masks = []
        self.dropout_p = dropout_p
        # We alternate masking: Evens vs Odds
        # This is a "Checkerboard-like" split for flattened vectors
        mask_even = torch.zeros(input_dim)
        mask_even[0::2] = 1 # [1, 0, 1, 0...]
        
        mask_odd = torch.zeros(input_dim)
        mask_odd[1::2] = 1  # [0, 1, 0, 1...]
        
        for i in range(num_layers):
            mask = mask_even if i % 2 == 0 else mask_odd
            self.masks.append(mask)
            
            self.layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=dropout_p),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=dropout_p),
                nn.Linear(hidden_dim, input_dim * 2) # s and t
            ))

    def forward(self, x):
        log_det_jac = 0
        z = x.view(-1, 784)
        
        for i, layer in enumerate(self.layers):
            mask = self.masks[i].to(x.device)
            masked_z = z * mask
            out = layer(masked_z)
            s, t = out.chunk(2, dim=1)
            
            # Tanh clamping is crucial for stability in high dim
            s = torch.tanh(s) * (1 - mask)
            t = t * (1 - mask)
            
            z = z * torch.exp(s) + t
            log_det_jac += s.sum(dim=1)
            
        return z, log_det_jac
    
    def inverse(self, z):
        x = z.view(-1, 784)
        
        for i, layer in reversed(list(enumerate(self.layers))):
            mask = self.masks[i].to(z.device)
            masked_x = x * mask
            out = layer(masked_x)
            s, t = out.chunk(2, dim=1)
            
            s = torch.tanh(s) * (1 - mask)
            t = t * (1 - mask)
            
            x = (x - t) * torch.exp(-s)
            
        x = x.view(-1, 1, 28, 28)
        return x
    

import torch
import torch.nn as nn



class GeneralFlow(nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()
        
        # 1. Convolutional Part (Locally smart)
        # We use fewer layers here since the Linear part will help
        self.conv_flow = ConvFlow(num_layers=4, hidden_dim=64, dropout_p=dropout_p)
        
        # 2. Linear Part (Globally smart)
        # Takes the flattened output of the Conv part
        self.linear_flow = SimpleFlow(input_dim=784, hidden_dim=1024, num_layers=4, dropout_p=dropout_p)
    def forward(self, x):
        log_det_total = 0
        
        # --- Stage 1: Convolutional Flow ---
        # Input: (N, 1, 28, 28)
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
            
        z, log_det_conv = self.conv_flow(x)
        log_det_total += log_det_conv
        
        # --- Stage 2: Flatten ---
        # Output of conv_flow is (N, 784) because we fixed the return shape previously
        # If your ConvFlow returns (N, 1, 28, 28), flatten it here:
        if z.dim() == 4:
            z = z.view(z.size(0), -1)
            
        # --- Stage 3: Linear Flow ---
        # Input: (N, 784)
        z, log_det_linear = self.linear_flow(z)
        log_det_total += log_det_linear
        
        return z, log_det_total

    def inverse(self, z):
        # We must reverse the order: Linear Inverse -> Unflatten -> Conv Inverse
        
        # 1. Linear Inverse
        z = self.linear_flow.inverse(z)
        
        # 2. Unflatten (handled inside ConvFlow.inverse usually, but let's be safe)
        if z.dim() == 2:
            z = z.view(-1, 1, 28, 28)
            
        # 3. Conv Inverse
        x = self.conv_flow.inverse(z)
        
        return x