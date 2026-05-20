import torch
import torch.nn as nn
import numpy as np

class GaussianPrior(nn.Module):
    def __init__(self, device, num_attr=2, total_dim=4704, independent_means=None, combinatorial_means=None):
        super().__init__()
        self.num_attr = num_attr
        self.total_dim = total_dim
        self.independent_means = independent_means.to(device) if independent_means is not None else None
        self.combinatorial_means = combinatorial_means.to(device) if combinatorial_means is not None else None

    def get_loss(self, z, sldj, labels):
        z_flat = z.view(z.shape[0], -1)
        nll = 0.5 * (z_flat ** 2).sum(dim=1) + 0.5 * z_flat.shape[1] * np.log(2 * np.pi)
        loss = (nll - sldj).mean()

        return loss

    def classify(self, z_flat):
        preds_ind = []
        complete_logits_ind = []
        preds_comb = []
        complete_logits_comb = []

        if self.independent_means is None or self.combinatorial_means is None:
            raise ValueError("One or more mean sets are not initialized")
        
        for i in range(self.num_attr):
            d = -((z_flat.unsqueeze(1) - self.independent_means[i].unsqueeze(0))**2).sum(2)
            preds_ind.append(d.argmax(1))
            complete_logits_ind.append(d)

        preds_ind = torch.stack(preds_ind, dim=1)
        complete_logits_ind = torch.stack(complete_logits_ind, dim=1)

        d = -((z_flat.unsqueeze(1) - self.combinatorial_means.unsqueeze(0))**2).sum(2)
        preds_comb = d.argmax(1)
        complete_logits_comb = d

        return preds_ind, complete_logits_ind, preds_comb, complete_logits_comb

    def get_full_latent(self, means):
        return torch.stack(means).sum(dim=0)