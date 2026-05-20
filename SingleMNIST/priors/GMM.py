import torch
import torch.nn as nn
import numpy as np

class GMM(nn.Module):
    def __init__(self, total_dim, num_classes, device, scale, fixed_means=False):
        super().__init__()

        if fixed_means:
            self.means = torch.randn(num_classes, total_dim, device=device) * scale
        else:
            self.means = nn.Parameter(torch.randn(num_classes, total_dim, device=device) * scale)
            self.means.requires_grad = True
            
    def get_loss(self, z, sldj, labels):
        y = labels
        z_flat = z.view(z.shape[0], -1)
        nll = 0.5 * ((z_flat - self.means[y]) ** 2).sum(dim=1) + 0.5 * z_flat.shape[1] * np.log(2 * np.pi)
        loss = (nll - sldj).mean()

        return loss

    def classify(self, z_flat):
        d = -((z_flat.unsqueeze(1) - self.means.unsqueeze(0))**2).sum(2)
        return d.argmax(1), d