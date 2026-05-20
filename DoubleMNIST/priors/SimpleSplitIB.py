import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSplitIB(nn.Module):
    def __init__(self, total_dim, arr_num_classes, beta, device, scale, fixed_means=False):
        super().__init__()

        self.num_attr = len(arr_num_classes)
        self.dims_per_attr = total_dim // self.num_attr
        self.remainder = total_dim % self.num_attr
        self.beta = beta
        if fixed_means:
            self.means = []
            for i in range(self.num_attr):
                dim_i = self.dims_per_attr + (1 if i < self.remainder else 0)
                mean = torch.randn(arr_num_classes[i], dim_i, device=device) * scale
                self.means.append(mean)
        else:
            self.means = nn.ParameterList()
            for i in range(self.num_attr):
                dim_i = self.dims_per_attr + (1 if i < self.remainder else 0)
                mean = nn.Parameter(torch.randn(arr_num_classes[i], dim_i, device=device) * scale)
                self.means.append(mean)

    def _bounds(self, i):
        start = i * self.dims_per_attr + min(i, self.remainder)
        end = start + self.dims_per_attr + (1 if i < self.remainder else 0)
        return start, end

    def get_loss(self, z, sldj, labels):
        y = labels
        z_flat = z.view(z.shape[0], -1)

        log_pz = 0
        loss_cls = 0

        for i in range(self.num_attr):
            start, end = self._bounds(i)
            z_i = z_flat[:, start:end]

            dists = -0.5 * ((z_i.unsqueeze(1) - self.means[i].unsqueeze(0)) ** 2).sum(2)

            log_pz += torch.logsumexp(dists, dim=1)
            loss_cls += F.cross_entropy(dists, y[:, i])

        loss_cls /= self.num_attr
        loss_gen = - (log_pz + sldj).mean() / z_flat.shape[1]

        return loss_gen + self.beta * loss_cls, loss_gen, loss_cls

    def classify(self, z_flat):
        preds = []
        complete_logits = []

        for i in range(self.num_attr):
            start, end = self._bounds(i)
            z_i = z_flat[:, start:end]

            d = -((z_i.unsqueeze(1) - self.means[i].unsqueeze(0)) ** 2).sum(2)
            preds.append(d.argmax(1))
            complete_logits.append(d)

        preds = torch.stack(preds, dim=1)
        return preds, complete_logits

    def get_full_latent(self, z_list):
        """
        z_list: list of vectors, one per attribute
        """
        return torch.cat(z_list, dim=1)

    def get_parts(self, z_flat):
        parts = []
        for i in range(self.num_attr):
            start, end = self._bounds(i)
            parts.append(z_flat[:, start:end])
        return parts
