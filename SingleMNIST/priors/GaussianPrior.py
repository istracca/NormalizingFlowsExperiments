import torch
import torch.nn as nn
import numpy as np

class GaussianPrior(nn.Module):
    def __init__(self, device, means=None, total_dim=None, num_classes=None):
        super().__init__()
        self.means = means.to(device) if means is not None else None
        self.total_dim = total_dim if total_dim is not None else (means.shape[1] if means is not None else None)
        self.num_classes = num_classes if num_classes is not None else (means.shape[0] if means is not None else None)

    def get_loss(self, z, sldj, labels=None):
        z_flat = z.view(z.shape[0], -1)
        nll = 0.5 * (z_flat ** 2).sum(dim=1) + 0.5 * z_flat.shape[1] * np.log(2 * np.pi)
        loss = (nll - sldj).mean()

        return loss

    def classify(self, z_flat):
        if self.means is None:
            raise ValueError("Means are not initialized")

        d = -((z_flat.unsqueeze(1) - self.means.unsqueeze(0))**2).sum(2)
        return d.argmax(1), d