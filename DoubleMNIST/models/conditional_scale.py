import torch
import torch.nn as nn
import torch.nn.functional as F

class Invertible1x1Conv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Initialize with a random orthogonal matrix
        w_init = torch.linalg.qr(torch.randn(channels, channels))[0]
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
        # x: (Batch, 1, 28, 56) -> (Batch, 4, 14, 28)
        b, c, h, w = x.shape
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 4, h // 2, w // 2)
        return x

    def inverse(self, x):
        # x: (Batch, 4, 14, 28) -> (Batch, 1, 28, 56)
        b, c, h, w = x.shape
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // 4, h * 2, w * 2)
        return x

class ConditionedCouplingNet(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, cond_dim, dropout_p=0.0):
        super().__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.cond_proj1 = nn.Linear(cond_dim, hidden_dim * 2)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=dropout_p)

        # Layer 2
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.cond_proj2 = nn.Linear(cond_dim, hidden_dim * 2)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=dropout_p)

        # Output Layer
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        
        # Zero initialization for stability
        nn.init.zeros_(self.cond_proj1.weight)
        nn.init.zeros_(self.cond_proj1.bias)
        nn.init.zeros_(self.cond_proj2.weight)
        nn.init.zeros_(self.cond_proj2.bias)
        self.conv3.weight.data.zero_()
        self.conv3.bias.data.zero_()

    def forward(self, x, c):
        # x: (B, C, H, W), c: (B, cond_dim)

        # Block 1
        h = self.conv1(x)
        gamma1, beta1 = self.cond_proj1(c).chunk(2, dim=1)
        gamma1 = gamma1.unsqueeze(2).unsqueeze(3)
        beta1 = beta1.unsqueeze(2).unsqueeze(3)
        
        # FiLM: Scale and Shift
        h = h * (1.0 + gamma1) + beta1
        h = self.drop1(self.relu1(self.bn1(h)))

        # Block 2
        h = self.conv2(h)
        gamma2, beta2 = self.cond_proj2(c).chunk(2, dim=1)
        gamma2 = gamma2.unsqueeze(2).unsqueeze(3)
        beta2 = beta2.unsqueeze(2).unsqueeze(3)
        
        # FiLM: Scale and Shift
        h = h * (1.0 + gamma2) + beta2
        h = self.drop2(self.relu2(self.bn2(h)))

        # Output
        return self.conv3(h)

class ChannelCouplingLayer(nn.Module):
    """
    Splits the tensor by channels, not by spatial mask.
    """
    def __init__(self, in_channels, cond_dim, hidden_dim=64, dropout_p=0.0):
        super().__init__()
        self.half_channels = in_channels // 2
        
        # Condition net: Takes 'half_channels', outputs 'half_channels * 2' (s and t)
        self.net = ConditionedCouplingNet(
            in_channels=self.half_channels,
            hidden_dim=hidden_dim,
            out_channels=self.half_channels * 2,
            cond_dim=cond_dim,
            dropout_p=dropout_p
        )

    def forward(self, x, c):
        # Split channels: x_a (active), x_p (passive/conditioner)
        x_a, x_p = x.chunk(2, dim=1)
        
        out = self.net(x_p, c)
        s, t = out.chunk(2, dim=1)
        
        s = torch.tanh(s)
        x_a = x_a * torch.exp(s) + t
        
        # Concatenate back
        return torch.cat([x_a, x_p], dim=1), s.sum(dim=[1, 2, 3])

    def inverse(self, x, c):
        x_a, x_p = x.chunk(2, dim=1)
        
        out = self.net(x_p, c)
        s, t = out.chunk(2, dim=1)
        
        s = torch.tanh(s)
        x_a = (x_a - t) * torch.exp(-s)
        
        return torch.cat([x_a, x_p], dim=1)

class ConditionedLinearNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, cond_dim, dropout_p=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.cond_proj1 = nn.Linear(cond_dim, hidden_dim * 2)
        self.act1 = nn.LeakyReLU(0.1)
        self.drop1 = nn.Dropout(p=dropout_p)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.cond_proj2 = nn.Linear(cond_dim, hidden_dim * 2)
        self.act2 = nn.LeakyReLU(0.1)
        self.drop2 = nn.Dropout(p=dropout_p)

        self.fc3 = nn.Linear(hidden_dim, out_dim)
        
        # Zero initialization for stability
        nn.init.zeros_(self.cond_proj1.weight)
        nn.init.zeros_(self.cond_proj1.bias)
        nn.init.zeros_(self.cond_proj2.weight)
        nn.init.zeros_(self.cond_proj2.bias)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()

    def forward(self, x, c):
        h = self.fc1(x)
        gamma1, beta1 = self.cond_proj1(c).chunk(2, dim=-1)
        
        # FiLM: Scale and Shift
        h = h * (1.0 + gamma1) + beta1 
        h = self.drop1(self.act1(h))
        
        h = self.fc2(h)
        gamma2, beta2 = self.cond_proj2(c).chunk(2, dim=-1)
        
        # FiLM: Scale and Shift
        h = h * (1.0 + gamma2) + beta2
        h = self.drop2(self.act2(h))
        
        return self.fc3(h)

class SimpleFlow(nn.Module):
    def __init__(self, cond_dim, input_dim=1568, hidden_dim=1024, num_layers=8, dropout_p=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.masks = []
        self.input_dim = input_dim
        
        mask_even = torch.zeros(input_dim)
        mask_even[0::2] = 1
        
        mask_odd = torch.zeros(input_dim)
        mask_odd[1::2] = 1
        
        for i in range(num_layers):
            mask = mask_even if i % 2 == 0 else mask_odd
            self.masks.append(mask)
            
            self.layers.append(ConditionedLinearNet(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                out_dim=input_dim * 2,
                cond_dim=cond_dim,
                dropout_p=dropout_p
            ))

    def forward(self, x, c):
        log_det_jac = 0
        z = x.view(-1, self.input_dim)
        
        for i, layer in enumerate(self.layers):
            mask = self.masks[i].to(x.device)
            masked_z = z * mask
            out = layer(masked_z, c)
            s, t = out.chunk(2, dim=1)
            
            s = torch.tanh(s) * (1 - mask)
            t = t * (1 - mask)
            
            z = z * torch.exp(s) + t
            log_det_jac += s.sum(dim=1)
            
        return z, log_det_jac
    
    def inverse(self, z, c):
        x = z.view(-1, self.input_dim)
        
        for i, layer in reversed(list(enumerate(self.layers))):
            mask = self.masks[i].to(z.device)
            masked_x = x * mask

            out = layer(masked_x, c)
            s, t = out.chunk(2, dim=1)
            
            s = torch.tanh(s) * (1 - mask)
            t = t * (1 - mask)
            
            x = (x - t) * torch.exp(-s)
            
        x = x.view(-1, 1, 28, 56)
        return x
    

class GeneralFlow(nn.Module):
    def __init__(self, num_classes=20, cond_dim=64, dropout_p=0.0):
        super().__init__()
        
        # --- CONDITIONING NETWORK ---
        # Extracts dense features from the one-hot class label
        self.cond_mlp = nn.Sequential(
            nn.Linear(num_classes, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        # --- SCALE 1: 1x28x56 -> 4x14x28 ---
        self.squeeze1 = Squeeze()
        self.flow1_couplings = nn.ModuleList()
        self.flow1_inv1x1 = nn.ModuleList()
        
        for _ in range(8):
            self.flow1_inv1x1.append(Invertible1x1Conv(channels=4))
            self.flow1_couplings.append(ChannelCouplingLayer(in_channels=4, cond_dim=cond_dim, hidden_dim=64, dropout_p=dropout_p))
        
        # --- SCALE 2: 4x14x28 -> 16x7x14 ---
        self.squeeze2 = Squeeze()
        self.flow2_couplings = nn.ModuleList()
        self.flow2_inv1x1 = nn.ModuleList()
        
        for _ in range(8):
            self.flow2_inv1x1.append(Invertible1x1Conv(channels=16))
            self.flow2_couplings.append(ChannelCouplingLayer(in_channels=16, cond_dim=cond_dim, hidden_dim=128, dropout_p=dropout_p))
        
        # --- Linear ---
        self.linear_flow = SimpleFlow(cond_dim=cond_dim, input_dim=1568, hidden_dim=2048, num_layers=4, dropout_p=dropout_p)

    def forward(self, x, y):
        # y is the two-hot class vector: (B, num_classes)
        c = self.cond_mlp(y)

        if x.dim() == 2: x = x.view(-1, 1, 28, 56)
            
        log_det_total = torch.zeros(x.size(0), device=x.device)
            
        # --- SCALE 1 ---
        x = self.squeeze1(x)
        
        for inv1x1, coupling in zip(self.flow1_inv1x1, self.flow1_couplings):
            # 1. Mix Channels (Learnable)
            x, ld_1x1 = inv1x1(x)
            log_det_total += ld_1x1
            
            # 2. Coupling
            x, ld_coup = coupling(x, c)
            log_det_total += ld_coup
            
        # --- SCALE 2 ---
        x = self.squeeze2(x)
        
        for inv1x1, coupling in zip(self.flow2_inv1x1, self.flow2_couplings):
            x, ld_1x1 = inv1x1(x)
            log_det_total += ld_1x1
            
            x, ld_coup = coupling(x, c)
            log_det_total += ld_coup
            
        # --- LINEAR ---
        x = x.view(x.size(0), -1)
        z, log_det_linear = self.linear_flow(x, c)
        log_det_total += log_det_linear
        
        return z, log_det_total

    def inverse(self, z, y):
        c = self.cond_mlp(y)
        
        z = self.linear_flow.inverse(z, c)
        z = z.view(-1, 16, 7, 14)
        
        # --- REVERSE SCALE 2 ---
        for inv1x1, coupling in zip(reversed(self.flow2_inv1x1), reversed(self.flow2_couplings)):
            z = coupling.inverse(z, c)
            z = inv1x1.inverse(z)
            
        z = self.squeeze2.inverse(z)
        
        # --- REVERSE SCALE 1 ---
        for inv1x1, coupling in zip(reversed(self.flow1_inv1x1), reversed(self.flow1_couplings)):
            z = coupling.inverse(z, c)
            z = inv1x1.inverse(z)
            
        x = self.squeeze1.inverse(z)
        return x