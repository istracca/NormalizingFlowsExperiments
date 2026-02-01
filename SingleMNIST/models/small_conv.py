import torch
import torch.nn as nn

class SimpleConvNet(nn.Module):
    """
    A lightweight Convolutional Network to predict s and t.
    It preserves spatial dimensions (padding=1 for 3x3 kernels).
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            # Input is 1 channel (grayscale), output hidden_dim
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1), # 1x1 conv mixing
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            # Output is 2 channels: one for s (scale), one for t (translation)
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1)
        )
        
        # Initialize the last layer to zeros is a common trick in Flows
        # so the flow starts as an identity function
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x):
        return self.net(x)

class GeneralFlow(nn.Module):
    def __init__(self, num_layers=8, hidden_dim=64):
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
            self.layers.append(SimpleConvNet(hidden_dim))

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