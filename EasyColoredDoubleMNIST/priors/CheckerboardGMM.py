import torch
import torch.nn as nn
import numpy as np

class CheckerboardGMM(nn.Module):
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
        y = labels # Shape: (batch, num_attr)
        z_flat = z.view(z.shape[0], -1)
        
        nll = 0
        for i in range(self.num_attr):
            # Checkerboard indexing: start at i, take every num_attr-th element
            # This yields a shape of (batch, dims_per_attr)
            z_subset = z_flat[:, i::self.num_attr] 
            
            # Select the correct mean for each sample in the batch based on label
            # self.means[i][y[:, i]] results in (batch, dims_per_attr)
            diff = z_subset - self.means[i][y[:, i]]
            
            nll += 0.5 * (diff ** 2).sum(dim=1) + 0.5 * self.dims_per_attr * np.log(2 * np.pi)
        
        loss = (nll - sldj).mean()

        return loss

    def classify(self, z_flat):
        preds = []
        complete_logits = []
        
        for i in range(self.num_attr):
            # Extract the i-th checkerboard component
            z_subset = z_flat[:, i::self.num_attr] # (batch, dims_per_attr)
            
            # Distance calculation: (batch, 1, dims) - (1, num_classes, dims)
            # Result: (batch, num_classes)
            sq_dist = -((z_subset.unsqueeze(1) - self.means[i].unsqueeze(0))**2).sum(2)
            
            preds.append(sq_dist.argmax(1))
            complete_logits.append(sq_dist)
        
        # Stack along dimension 1 to get shape (batch_size, num_attr)
        preds = torch.stack(preds, dim=1) 
        return preds, complete_logits
    
    def get_full_latent(self, z_list, temp=0.0):
        """
        z_list: list of vectors, one per attribute
        """
        z_flat = torch.zeros(z_list[0].size(0), self.num_attr * self.dims_per_attr, device=z_list[0].device)
        for i in range(self.num_attr):
            z_flat[:, i::self.num_attr] = z_list[i]
        return z_flat

    def get_parts(self, z_flat):
        parts = []
        for i in range(self.num_attr):
            parts.append(z_flat[:, i::self.num_attr])
        return parts