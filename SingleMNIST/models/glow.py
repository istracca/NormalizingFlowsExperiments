import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import special_ortho_group

from math import exp

class GlowCouplingBlock(nn.Module):

    def __init__(self, dims_in, dims_c=[], subnet_constructor=None, clamp=2.):
        super().__init__()

        channels = dims_in[0][0]
        if dims_c:
            raise ValueError('does not support conditioning yet')

        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2
        self.splits = [self.split_len1, self.split_len2]

        self.in_channels = channels
        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.conditional = False

        self.s1 = subnet_constructor(self.split_len1, 2 * self.split_len2)
        self.s2 = subnet_constructor(self.split_len2, 2 * self.split_len1)

        self.last_jac = None

    def log_e(self, s):
        return self.clamp * torch.tanh(0.2 * s)

    def affine(self, x, a, rev=False):
        ch = x.shape[1]
        sub_jac = self.log_e(a[:,:ch])
        if not rev:
            return (x * torch.exp(sub_jac) + a[:,ch:],
                    torch.sum(sub_jac, dim=(1,2,3)))
        else:
            return ((x - a[:,ch:]) * torch.exp(-sub_jac),
                    -torch.sum(sub_jac, dim=(1,2,3)))

    def forward(self, x, c=[], rev=False):
        x1, x2 = torch.split(x[0], self.splits, dim=1)

        if not rev:
            a1 = self.s1(x1)
            y2, j2 = self.affine(x2, a1)

            a2 = self.s2(y2)
            y1, j1 = self.affine(x1, a2)

        else: # names of x and y are swapped!
            a2 = self.s2(x2)
            y1, j1 = self.affine(x1, a2, rev=True)

            a1 = self.s1(y1)
            y2, j2 = self.affine(x2, a1, rev=True)

        self.last_jac = j1 + j2
        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, c=[], rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        return input_dims


class AIO_GlowCouplingBlock(GlowCouplingBlock):

    def __init__(self,
        dims_in,
        dims_c=[],
        subnet_constructor=None,
        clamp=2.,
        act_norm=1.,
        act_norm_type='SOFTPLUS',
        permute_soft=False
    ):

        super().__init__(dims_in, dims_c=dims_c, subnet_constructor=subnet_constructor, clamp=clamp)

        if act_norm_type == 'SIGMOID':
            act_norm = np.log(act_norm)
            self.actnorm_activation = (lambda a: 10 * torch.sigmoid(a - 2.))
        elif act_norm_type == 'SOFTPLUS':
            act_norm = 10. * act_norm
            self.softplus = nn.Softplus(beta=0.5)
            self.actnorm_activation = (lambda a: 0.1 * self.softplus(a))
        elif act_norm_type == 'EXP':
            act_norm = np.log(act_norm)
            self.actnorm_activation = (lambda a: torch.exp(a))
        else:
            raise ValueError('Please, SIGMOID, SOFTPLUS or EXP, as actnorm type')

        assert act_norm > 0., "please, this is not allowed. don't do it. take it... and go."
        channels = self.in_channels

        self.act_norm = nn.Parameter(torch.ones(1, channels, 1, 1) * float(act_norm))
        self.act_offset = nn.Parameter(torch.zeros(1, channels, 1, 1))

        if permute_soft:
            w = special_ortho_group.rvs(channels)
        else:
            w = np.zeros((channels,channels))
            for i,j in enumerate(np.random.permutation(channels)):
                w[i,j] = 1.
        w_inv = w.T

        self.w = nn.Parameter(torch.FloatTensor(w).view(channels, channels, 1, 1), requires_grad=False)
        self.w_inv = nn.Parameter(torch.FloatTensor(w_inv).view(channels, channels, 1, 1), requires_grad=False)

    def permute(self, x, rev=False):
        scale = self.actnorm_activation( self.act_norm)
        if rev:
            return (F.conv2d(x, self.w_inv) - self.act_offset) / scale
        else:
            return F.conv2d(x * scale + self.act_offset, self.w)

    def forward(self, x, c=[], rev=False):
        if rev:
            x = [self.permute(x[0], rev=True)]

        x_out = super().forward(x, c=[], rev=rev)[0]

        if not rev:
            x_out = self.permute(x_out, rev=False)

        n_pixels = x_out.shape[2] * x_out.shape[3]
        self.last_jac += ((-1)**rev * n_pixels) * (torch.log(self.actnorm_activation(self.act_norm) + 1e-12).sum())
        return [x_out]

# --- 1. SOTTO-RETI E UTILS ---
def zero_init(module):
    """Inizializza l'ultimo layer a 0 per stabilità iniziale."""
    module.weight.data.zero_()
    module.bias.data.zero_()
    return module

def subnet_conv(dims_in, dims_out):
    """La piccola CNN dentro ogni blocco invertibile."""
    return nn.Sequential(
        nn.Conv2d(dims_in, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=1),
        nn.ReLU(),
        zero_init(nn.Conv2d(128, dims_out, kernel_size=3, padding=1))
    )

class SqueezeLayer(nn.Module):
    """Trasforma [B, 1, 28, 28] -> [B, 4, 14, 14] per dare canali ai Coupling Layer."""
    def __init__(self, factor=2):
        super().__init__()
        self.factor = factor

    def forward(self, x_list, rev=False):
        x = x_list[0]
        b, c, h, w = x.shape

        if not rev: # Forward (Squeeze): 28x28 -> 14x14 (canali x4)
            x = x.view(b, c, h // self.factor, self.factor, w // self.factor, self.factor)
            x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
            x = x.view(b, c * self.factor * self.factor, h // self.factor, w // self.factor)
            return [x]
            
        else: # Inverse (Unsqueeze): 14x14 (canali /4) -> 28x28
            # --- CORREZIONE QUI ---
            # Prima dividiamo i canali nei blocchi (factor x factor).
            # NON moltiplichiamo h e w qui, perché h e w sono ancora 14!
            x = x.view(b, c // (self.factor**2), self.factor, self.factor, h, w)
            
            # Permutiamo per rimettere i blocchi al posto giusto spazialmente
            x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
            
            # Ora uniamo le dimensioni per ottenere l'immagine grande (28x28)
            x = x.view(b, c // (self.factor**2), h * self.factor, w * self.factor)
            return [x]

    def jacobian(self, x, c=[], rev=False):
        return 0.0

# --- 2. ARCHITETTURA FLOW (Il Modello) ---
class GeneralFlow(nn.Module):
    def __init__(self, num_blocks=8, input_channels=1):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Squeeze iniziale: aumenta canali da 1 a 4
        self.layers.append(SqueezeLayer(factor=2))
        
        current_channels = input_channels * 4 
        h, w = 14, 14 
        
        # Blocchi Glow in sequenza
        for i in range(num_blocks):
            self.layers.append(
                AIO_GlowCouplingBlock(
                    dims_in=[(current_channels, h, w)],
                    subnet_constructor=subnet_conv,
                    clamp=2.0,
                    permute_soft=True # Impara a mescolare i canali
                )
            )
            
    def forward(self, x):
        """Forward: Immagine -> Z"""
        x_list = [x]
        total_jac = 0.0
        
        for layer in self.layers:
            x_list = layer(x_list, rev=False)
            if hasattr(layer, 'last_jac') and layer.last_jac is not None:
                total_jac += layer.last_jac
            elif hasattr(layer, 'jacobian'):
                total_jac += layer.jacobian(x_list, rev=False)
                
        return x_list[0], total_jac

    def inverse(self, z):
        """Inverse: Z -> Immagine"""
        x_list = [z]
        for layer in reversed(self.layers):
            x_list = layer(x_list, rev=True)
        return x_list[0]