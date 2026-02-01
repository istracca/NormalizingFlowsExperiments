import torch
import torch.nn as nn
import torch.nn.functional as F

class Invertible1x1Conv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Initialize with a random orthogonal matrix
        # This keeps the training stable at the start
        w_init = torch.linalg.qr(torch.randn(channels, channels))[0]
        
        # Make it a parameter so we can learn it
        # Shape: (C, C, 1, 1) for Conv2d
        self.weight = nn.Parameter(w_init.unsqueeze(2).unsqueeze(3))

    def forward(self, x):
        # x: (B, C, H, W)
        
        # 1. Calculate Log Determinant
        # We only need the determinant of the (C, C) matrix, not the spatial part
        # slogdet returns (sign, logabsdet)
        # We squeeze the spatial dims to get the CxC matrix
        w_matrix = self.weight.squeeze().float()
        _, log_abs_det = torch.slogdet(w_matrix)
        
        # The total log_det is sum over all spatial pixels (H * W)
        b, c, h, w = x.shape
        log_det = h * w * log_abs_det
        
        # 2. Apply Convolution (Matrix Multiplication)
        z = F.conv2d(x, self.weight)
        
        return z, log_det

    def inverse(self, z):
        # 1. Compute Inverse Matrix
        w_matrix = self.weight.squeeze().float()
        w_inv = torch.inverse(w_matrix)
        
        # Reshape for Conv2d: (C, C, 1, 1)
        w_inv = w_inv.unsqueeze(2).unsqueeze(3)
        
        # 2. Apply Inverse Convolution
        x = F.conv2d(z, w_inv)
        return x

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


class SimpleFlow(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=1024, num_layers=8, dropout_p=0.0):
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
    

class GeneralFlow(nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()
        
        # --- SCALE 1: 28x28 -> 14x14 ---
        self.squeeze1 = Squeeze()
        
        # 4 Layers at Scale 1
        self.flow1_couplings = nn.ModuleList()
        self.flow1_inv1x1 = nn.ModuleList()
        
        for _ in range(8):
            # We add an Invertible1x1Conv before every Coupling layer
            self.flow1_inv1x1.append(Invertible1x1Conv(channels=4))
            self.flow1_couplings.append(ChannelCouplingLayer(in_channels=4, hidden_dim=64, dropout_p=dropout_p))
        
        # --- SCALE 2: 14x14 -> 7x7 ---
        self.squeeze2 = Squeeze()
        
        # 4 Layers at Scale 2
        self.flow2_couplings = nn.ModuleList()
        self.flow2_inv1x1 = nn.ModuleList()
        
        for _ in range(8):
            # Channels are 16 now
            self.flow2_inv1x1.append(Invertible1x1Conv(channels=16))
            self.flow2_couplings.append(ChannelCouplingLayer(in_channels=16, hidden_dim=128, dropout_p=dropout_p))
        
        # --- GLOBAL: Linear ---
        # Flatten input: 16 * 7 * 7 = 784
        self.linear_flow = SimpleFlow(input_dim=784, hidden_dim=2048, num_layers=4, dropout_p=dropout_p)

    def forward(self, x):
        # Ensure 4D input
        if x.dim() == 2: x = x.view(-1, 1, 28, 28)
            
        # --- FIX: Initialize log_det as a vector, not a scalar ---
        # Shape: (Batch_Size,)
        log_det_total = torch.zeros(x.size(0), device=x.device)
            
        # --- SCALE 1 ---
        x = self.squeeze1(x)
        
        for inv1x1, coupling in zip(self.flow1_inv1x1, self.flow1_couplings):
            # 1. Mix Channels (Learnable)
            x, ld_1x1 = inv1x1(x)
            # ld_1x1 is a scalar, but adding it to a vector (log_det_total) is safe
            log_det_total += ld_1x1
            
            # 2. Coupling
            x, ld_coup = coupling(x)
            # ld_coup is a vector (Batch,), now this addition works perfectly
            log_det_total += ld_coup
            
        # --- SCALE 2 ---
        x = self.squeeze2(x)
        
        for inv1x1, coupling in zip(self.flow2_inv1x1, self.flow2_couplings):
            x, ld_1x1 = inv1x1(x)
            log_det_total += ld_1x1
            
            x, ld_coup = coupling(x)
            log_det_total += ld_coup
            
        # --- LINEAR ---
        x = x.view(x.size(0), -1)
        z, log_det_linear = self.linear_flow(x)
        log_det_total += log_det_linear
        
        return z, log_det_total

    def inverse(self, z):
        z = self.linear_flow.inverse(z)
        z = z.view(-1, 16, 7, 7)
        
        # --- REVERSE SCALE 2 ---
        # Note: We must iterate in reverse and apply operations in reverse order
        # Forward: Inv1x1 -> Coupling
        # Inverse: Coupling_Inv -> Inv1x1_Inv
        
        for inv1x1, coupling in zip(reversed(self.flow2_inv1x1), reversed(self.flow2_couplings)):
            z = coupling.inverse(z)
            z = inv1x1.inverse(z)
            
        z = self.squeeze2.inverse(z)
        
        # --- REVERSE SCALE 1 ---
        for inv1x1, coupling in zip(reversed(self.flow1_inv1x1), reversed(self.flow1_couplings)):
            z = coupling.inverse(z)
            z = inv1x1.inverse(z)
            
        x = self.squeeze1.inverse(z)
        return x