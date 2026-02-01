import torch
import torch.nn as nn

class Squeeze(nn.Module):
    def forward(self, x):
        # x: (Batch, 1, 28, 28) -> (Batch, 4, 14, 14)
        b, c, h, w = x.shape
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 4, h // 2, w // 2)
        return x

    def inverse(self, x):
        # x: (Batch, 4, 14, 14) -> (Batch, 1, 28, 28)
        b, c, h, w = x.shape
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // 4, h * 2, w * 2)
        return x

class ChannelCouplingLayer(nn.Module):
    """
    Splits the tensor by CHANNELS, not by spatial mask.
    This eliminates checkerboard artifacts.
    """
    def __init__(self, in_channels, hidden_dim=64, dropout_p=0.0):
        super().__init__()
        self.half_channels = in_channels // 2
        
        # Condition net: Takes 'half_channels', outputs 'half_channels * 2' (s and t)
        self.net = nn.Sequential(
            nn.Conv2d(self.half_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1), # 1x1 conv mixing
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(hidden_dim, self.half_channels * 2, kernel_size=3, padding=1)
        )
        # Zero init for identity start
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x):
        # Split channels: x_a (active), x_p (passive/conditioner)
        x_a, x_p = x.chunk(2, dim=1)
        
        out = self.net(x_p)
        s, t = out.chunk(2, dim=1)
        
        s = torch.tanh(s)
        x_a = x_a * torch.exp(s) + t
        
        # Concatenate back
        return torch.cat([x_a, x_p], dim=1), s.sum(dim=[1, 2, 3])

    def inverse(self, x):
        x_a, x_p = x.chunk(2, dim=1)
        
        out = self.net(x_p)
        s, t = out.chunk(2, dim=1)
        
        s = torch.tanh(s)
        x_a = (x_a - t) * torch.exp(-s)
        
        return torch.cat([x_a, x_p], dim=1)

class SqueezeFlow(nn.Module):
    def __init__(self, num_layers=4, hidden_dim=64, dropout_p=0.0):
        super().__init__()
        self.squeeze = Squeeze()
        self.layers = nn.ModuleList()
        
        # We prefer even number of layers so we can swap the channel order
        # to ensure all channels get processed.
        for i in range(num_layers):
            self.layers.append(ChannelCouplingLayer(in_channels=4, hidden_dim=hidden_dim, dropout_p=dropout_p))

    def forward(self, x):
        log_det_total = 0
        
        # 1. Squeeze: (N, 1, 28, 28) -> (N, 4, 14, 14)
        x = self.squeeze(x)
        
        for i, layer in enumerate(self.layers):
            x, log_det = layer(x)
            log_det_total += log_det
            
            # Flip channels after every layer so the "passive" part 
            # becomes "active" in the next layer.
            # Equivalent to alternating masks.
            x = x.flip(dims=(1,)) 
            
        return x, log_det_total

    def inverse(self, z):
        # Reverse the loop
        for i, layer in reversed(list(enumerate(self.layers))):
            # We must flip first because we flipped at the END of forward
            z = z.flip(dims=(1,))
            z = layer.inverse(z)
            
        x = self.squeeze.inverse(z)
        return x
    

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFlow(nn.Module):
    def __init__(self, height, width, hidden_dim=1024, num_layers=8, dropout_p=0.0):
        super().__init__()
        self.height = height
        self.width = width
        self.layers = nn.ModuleList()
        self.masks = []
        
        # We alternate masking: Evens vs Odds
        # This is a "Checkerboard-like" split for flattened vectors
        mask_even = torch.zeros(self.height * self.width)
        mask_even[0::2] = 1 # [1, 0, 1, 0...]
        
        mask_odd = torch.zeros(self.height * self.width)
        mask_odd[1::2] = 1  # [0, 1, 0, 1...]
        
        for i in range(num_layers):
            mask = mask_even if i % 2 == 0 else mask_odd
            self.masks.append(mask)
            
            self.layers.append(nn.Sequential(
                nn.Linear(self.height * self.width, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=dropout_p),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=dropout_p),
                nn.Linear(hidden_dim, self.height * self.width * 2) # s and t
            ))

    def forward(self, x):
        log_det_jac = 0
        z = x.view(-1, self.height * self.width)
        
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
        x = z.view(-1, self.height * self.width)
        
        for i, layer in reversed(list(enumerate(self.layers))):
            mask = self.masks[i].to(z.device)
            masked_x = x * mask
            out = layer(masked_x)
            s, t = out.chunk(2, dim=1)
            
            s = torch.tanh(s) * (1 - mask)
            t = t * (1 - mask)
            
            x = (x - t) * torch.exp(-s)
            
        x = x.view(-1, 1, self.height, self.width)
        return x
    

import torch
import torch.nn as nn



class GeneralFlow(nn.Module):
    def __init__(self, dropout_p=0.0, height=28, width=56):
        super().__init__()
        self.height = height
        self.width = width
        
        # 1. Convolutional Part (Locally smart)
        # We use fewer layers here since the Linear part will help
        self.squeeze_flow = SqueezeFlow(num_layers=8, hidden_dim=64, dropout_p=dropout_p)
        
        # 2. Linear Part (Globally smart)
        # Takes the flattened output of the Conv part
        self.linear_flow = SimpleFlow(height=self.height, width=self.width, hidden_dim=1024, num_layers=4, dropout_p=dropout_p)
    def forward(self, x):
        log_det_total = 0
        
        # --- Stage 1: Convolutional Flow ---
        # Input: (N, 1, 28, 56)
        if x.dim() == 2:
            x = x.view(-1, 1, self.height, self.width)
            
        z, log_det_conv = self.squeeze_flow(x)
        log_det_total += log_det_conv
        
        # --- Stage 2: Flatten ---
        # Output of squeeze_flow is (N, 1568) because we fixed the return shape previously
        # If your ConvFlow returns (N, 1, 28, 56), flatten it here:
        if z.dim() == 4:
            z = z.view(z.size(0), -1)
            
        # --- Stage 3: Linear Flow ---
        # Input: (N, 1568)
        z, log_det_linear = self.linear_flow(z)
        log_det_total += log_det_linear
        
        return z, log_det_total

    def inverse(self, z):
        # We must reverse the order: Linear Inverse -> Unflatten -> Conv Inverse
        
        # 1. Linear Inverse
        z = self.linear_flow.inverse(z)
        
        # 2. Unflatten (handled inside ConvFlow.inverse usually, but let's be safe)
        z_spatial = z.view(-1, 4, self.height // 2, self.width // 2)
            
        # 3. Conv Inverse
        x = self.squeeze_flow.inverse(z_spatial)
        
        return x