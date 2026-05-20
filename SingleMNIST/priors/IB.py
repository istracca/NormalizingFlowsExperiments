import torch
import torch.nn as nn
import torch.nn.functional as F

class IB(nn.Module):
    def __init__(self, total_dim, num_classes, beta, device, scale, fixed_means=False):
        super().__init__()
        self.beta = beta
        if fixed_means:
            self.means = torch.randn(num_classes, total_dim, device=device) * scale
        else:
            self.means = nn.Parameter(torch.randn(num_classes, total_dim, device=device) * scale)
            self.means.requires_grad = True
            
 
    def get_loss(self, z, sldj, labels):
        y = labels
        z_flat = z.view(z.shape[0], -1)
        
        dists = -0.5 * ((z_flat.unsqueeze(1) - self.means.unsqueeze(0))**2).sum(2)
        loss_cls = F.cross_entropy(dists, y)
        
        log_pz = torch.logsumexp(dists, dim=1)
        loss_gen = - (log_pz + sldj).mean() / z_flat.shape[1]
        
        return loss_gen + self.beta * loss_cls, loss_gen, loss_cls

    def classify(self, z_flat):
        d = -((z_flat.unsqueeze(1) - self.means.unsqueeze(0))**2).sum(2)
        return d.argmax(1), d