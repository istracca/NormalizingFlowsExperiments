import torch
import torch.nn as nn
import numpy as np

class SimpleSplitGMM(nn.Module):
    def __init__(self, total_dim, arr_num_classes, device, scale, fixed_means=True):
        super().__init__()

        self.num_attr = len(arr_num_classes)
        dims_per_attr = total_dim // self.num_attr
        self.dims_per_attr = dims_per_attr
        self.means = []

        if fixed_means:
            for i in range(self.num_attr):
                mean = torch.randn(arr_num_classes[i], dims_per_attr, device=device) * scale
                self.means.append(mean)
            
        else:
            for i in range(self.num_attr):
                mean = nn.Parameter(torch.randn(arr_num_classes[i], dims_per_attr, device=device) * scale)
                mean.requires_grad = True
                self.means.append(mean)

    def get_loss(self, z, sldj, labels):
        y = labels
        z_flat = z.view(z.shape[0], -1)
        nll = 0
        for i in range(self.num_attr):
            start = i * self.dims_per_attr
            end = (i + 1) * self.dims_per_attr
            nll += 0.5 * ((z_flat[:, start:end] - self.means[i][y[:, i]]) ** 2).sum(dim=1) + 0.5 * self.dims_per_attr * np.log(2 * np.pi)
    
        loss = (nll - sldj).mean()
        return loss

    def classify(self, z_flat):
        chunks = [z_flat[:, i * self.dims_per_attr : (i + 1) * self.dims_per_attr] for i in range(self.num_attr)]
        preds = []
        complete_logits = []
        for i in range(self.num_attr):
            d = -((chunks[i].unsqueeze(1) - self.means[i].unsqueeze(0))**2).sum(2)
            preds.append(d.argmax(1))
            complete_logits.append(d)
        
        preds = torch.stack(preds, dim=1) 
        return preds, complete_logits
    
    def get_full_latent(self, z_list):
        """
        z_list: list of vectors, one per attribute
        """
        z = torch.cat(z_list, dim=1)
        return z
    
    def get_parts(self, z_flat):
        parts = []
        for i in range(self.num_attr):
            start = i * self.dims_per_attr
            end = (i + 1) * self.dims_per_attr
            parts.append(z_flat[:, start:end])
        return parts