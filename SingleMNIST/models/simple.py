import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralFlow(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=1024, num_layers=8):
        super().__init__()
        self.layers = nn.ModuleList()
        self.masks = []
        
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
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.1),
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