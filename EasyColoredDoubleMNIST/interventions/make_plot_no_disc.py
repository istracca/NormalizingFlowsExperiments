import os
import sys
import torch
import random
import importlib
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join('..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../priors')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

from utils import set_seed
from CheckerboardGMM import CheckerboardGMM
from CheckerboardIB import CheckerboardIB
from GaussianPrior import GaussianPrior

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VERSION = '2_attr'
ARR_NUM_CLASSES = [10, 10]

SCALES = [1.0,2.0,3.0,4.0,5.0]
BETAS = [0.01, 0.05, 0.1]
LAMBDAS_PMF = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
LAMBDAS_GAUSS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

BASE_COLORS = torch.tensor([
    [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]
], dtype=torch.float32).to(DEVICE)

def get_dominant_colors(imgs, device):
    imgs_shifted = imgs + 0.5 
    left_half = imgs_shifted[:, :, :, :28]
    right_half = imgs_shifted[:, :, :, 28:]
    
    mask_left = (left_half.max(dim=1, keepdim=True)[0] > 0.2).float()
    mask_right = (right_half.max(dim=1, keepdim=True)[0] > 0.2).float()
    
    mean_left = (left_half * mask_left).sum(dim=(2, 3)) / (mask_left.sum(dim=(2, 3)) + 1e-5)
    mean_right = (right_half * mask_right).sum(dim=(2, 3)) / (mask_right.sum(dim=(2, 3)) + 1e-5)
    
    color_left = torch.argmin(torch.cdist(mean_left, BASE_COLORS), dim=1)
    color_right = torch.argmin(torch.cdist(mean_right, BASE_COLORS), dim=1)
    return color_left, color_right

def parse_predictions(preds_raw):
    if isinstance(preds_raw, list): return torch.stack(preds_raw, dim=1)
    elif isinstance(preds_raw, tuple): return torch.stack(preds_raw[0], dim=1) if isinstance(preds_raw[0], list) else preds_raw[0]
    return preds_raw

def collect_pmf_samples(loader, disc_model, model_type="GMM"):
    print(f"Collecting PMF ({model_type}) Samples (CC Failure, Style Success)...")
    pool = []
    gen_module = importlib.import_module("hybrid_v3_1x1_double")
    
    for param in (SCALES if model_type == "GMM" else BETAS):
        gen = gen_module.GeneralFlow(dropout_p=0.1).to(DEVICE)
        
        if model_type == "GMM":
            prior = CheckerboardGMM(total_dim=4704, arr_num_classes=[10, 10], device=DEVICE, scale=param, fixed_means=True)
            path = f"../experiments/models/GMM/2_attr/best_loss_{param}_hybrid_v3_1x1_double_CheckerboardGMM_Adam_0.5_0.1.pth"
        else:
            prior = CheckerboardIB(total_dim=4704, arr_num_classes=[10, 10], beta=param, device=DEVICE, scale=0.0, fixed_means=False)
            path = f"../experiments/models/IB/2_attr/best_loss_0.0_hybrid_v3_1x1_double_CheckerboardIB_{param}_Adam_0.5_0.1_False_20_.pth"
            
        prior.num_classes = 20; prior.num_attr = 2; prior.total_dim = 4704
        ckpt = torch.load(path, map_location=DEVICE)
        gen.load_state_dict(ckpt['model_state_dict']); prior.load_state_dict(ckpt['prior_state_dict']); prior.means = ckpt['means']
        gen.eval(); prior.eval()

        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X_centered = (((batch_X.to(DEVICE) * 255.0 + torch.rand_like(batch_X.to(DEVICE))) / 256.0) - 0.5)
                orig_c_left, orig_c_right = get_dominant_colors(batch_X_centered, DEVICE)
                z, _ = gen(batch_X_centered)
                
                preds_pre = parse_predictions(prior.classify(z.view(z.size(0), -1))[0])
                left_z, right_z = prior.get_parts(z.view(z.size(0), -1))[0], prior.get_parts(z.view(z.size(0), -1))[1]
                mu_left_orig, mu_right_orig = prior.means[0][preds_pre[:, 0]].to(DEVICE), prior.means[1][preds_pre[:, 1]].to(DEVICE)
                
                for int_type in ['attr0', 'attr1', 'both']:
                    t0, t1 = preds_pre[:, 0].clone(), preds_pre[:, 1].clone()
                    if int_type in ['attr0', 'both']: t0 = (t0 + random.randint(1, 9)) % 10
                    if int_type in ['attr1', 'both']: t1 = (t1 + random.randint(1, 9)) % 10
                    
                    mu_left_target, mu_right_target = prior.means[0][t0].to(DEVICE), prior.means[1][t1].to(DEVICE)
                    
                    for lam in LAMBDAS_PMF:
                        left_z_shifted = mu_left_target + lam * (left_z - mu_left_orig)
                        right_z_shifted = mu_right_target + lam * (right_z - mu_right_orig)
                        
                        if int_type == "attr0": z_interv = prior.get_full_latent([left_z_shifted, right_z])
                        elif int_type == "attr1": z_interv = prior.get_full_latent([left_z, right_z_shifted])
                        else: z_interv = prior.get_full_latent([left_z_shifted, right_z_shifted])

                        img_interv = gen.inverse(z_interv.view(z.size(0), 12, 14, 28))
                        preds_post_prior = parse_predictions(prior.classify(z_interv.view(z.size(0), -1))[0])
                        preds_post_disc = torch.stack([torch.argmax(logits, dim=1) for logits in disc_model(img_interv)], dim=1)
                        c_left, c_right = get_dominant_colors(img_interv, DEVICE)
                        
                        succ = (preds_post_prior[:, 0] == t0) & (preds_post_prior[:, 1] == t1) & ((preds_post_disc[:, 1] != t1) | (preds_post_disc[:, 1] != t1)) & ((c_left == orig_c_left) & (c_right == orig_c_right))
                        for idx in torch.nonzero(succ, as_tuple=False).squeeze(-1):
                            pool.append({
                                'img_orig': batch_X[idx].cpu().numpy(), 'img_interv': img_interv[idx].cpu().numpy(),
                                'orig_classes': [preds_pre[idx, 0].item(), preds_pre[idx, 1].item()],
                                'tgt_classes': [t0[idx].item(), t1[idx].item()],
                                'pred_post_prior': [preds_post_prior[idx, 0].item(), preds_post_prior[idx, 1].item()],
                                'pred_post_disc': [preds_post_disc[idx, 0].item(), preds_post_disc[idx, 1].item()],
                                'lambda': lam, 'scale' if model_type == "GMM" else 'beta': param, 'int_type': int_type
                            })
                if len(pool) >= 50: break
        del gen, prior; torch.cuda.empty_cache()
    return pool

def filter_balanced_samples(pool, n_samples, param_key=None):
    """Filters pool to ensure uniqueness and guarantees 1/3 attr0, 1/3 attr1, 1/3 both."""
    n_attr0 = n_samples // 3
    n_attr1 = n_samples // 3
    n_both = n_samples - n_attr0 - n_attr1                                                        
    
    targets = {'attr0': n_attr0, 'attr1': n_attr1, 'both': n_both}
    counts = {'attr0': 0, 'attr1': 0, 'both': 0}
    
    selected = []; seen_hashes = set(); seen_params = set()
    random.shuffle(pool)
    
    for item in pool:
        itype = item['int_type']
        if counts[itype] >= targets[itype]: 
            continue
            
        orig_hash = hash(item['img_orig'].tobytes())
        if orig_hash in seen_hashes: 
            continue
            
        if param_key:
            combo = (item['lambda'], item[param_key])
            if combo in seen_params: 
                continue
            seen_params.add(combo)
            
        seen_hashes.add(orig_hash)
        counts[itype] += 1
        selected.append(item)
        
        if len(selected) >= n_samples: break
        
    return selected

def plot_comprehensive_interventions(data, save_path="plots/comprehensive_interventions_no_disc.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = plt.figure(figsize=(19, 20), layout="constrained")
    main_subfigs = fig.subfigures(2, 1, height_ratios=[1,0.001])
    
    def populate_axes(axes, data_list, section_type):
        axes = axes.flatten()
        for ax, item in zip(axes, data_list):
            orig_rgb = np.clip(item['img_orig'].transpose(1, 2, 0), 0, 1)
            interv_rgb = np.clip(item['img_interv'].transpose(1, 2, 0) + 0.5, 0, 1)
            ax.imshow(np.concatenate([orig_rgb, interv_rgb], axis=1))
            ax.axvline(x=orig_rgb.shape[1], color='lightgray', linestyle='--', linewidth=1.5)
            ax.axis('off')
            
            o_c = item['orig_classes']
            t_c = item['tgt_classes']
            p_p = item['pred_post_prior']
            p_d = item['pred_post_disc']
            class_str = f"[{o_c[0]}, {o_c[1]}] -> [{t_c[0]}, {t_c[1]}] \n PMF: [{p_p[0]}, {p_p[1]}] Disc: [{p_d[0]}, {p_d[1]}]"
            
            if section_type == 'GMM':
                ax.set_title(class_str + "\n" + r"$s={} \mid \lambda={}$".format(item['scale'], item['lambda']), fontsize=20)
            elif section_type == 'IB':
                ax.set_title(class_str + "\n" + r"$\beta={} \mid \lambda={}$".format(item['beta'], item['lambda']), fontsize=20)
            else:
                ax.set_title(class_str, fontsize=20)

    main_subfigs[0].suptitle("PMF", fontsize=32, fontweight='bold')
    pmf_subfigs = main_subfigs[0].subfigures(2, 1)
    
    pmf_subfigs[0].suptitle("GMM", fontsize=28)
    populate_axes(pmf_subfigs[0].subplots(4, 4, gridspec_kw={'hspace': 0.0}), data['GMM'], 'GMM')
    
    pmf_subfigs[1].suptitle("IB", fontsize=28)
    populate_axes(pmf_subfigs[1].subplots(4, 4, gridspec_kw={'hspace': 0.0}), data['IB'], 'IB')

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"\nPipeline Complete. Plot successfully saved to {save_path}")


if __name__ == "__main__":
    set_seed(42)
    
    print("Loading test dataset...")
    data_np = np.load('../data/easy_colored_double_mnist.npz')
    X_test_tensor = torch.tensor(data_np['X_test'].transpose(0, 3, 1, 2), dtype=torch.float32)
    y_test_tensor = torch.tensor(data_np['y_test'][:, 0:2], dtype=torch.long)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=4096, shuffle=False)

    X_train_tensor = torch.tensor(data_np['X_train'].transpose(0, 3, 1, 2), dtype=torch.float32)
    y_train_tensor = torch.tensor(data_np['y_train'][:, 0:2], dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=4096, shuffle=False)

    print("Loading Global Discriminator...")
    disc_module = importlib.import_module("disc_v3_double")
    disc_model = disc_module.PseudoResNet(arr_num_classes=[10, 10], in_channels=3, dropout_p=0.1, device=DEVICE).to(DEVICE)
    disc_model.load_state_dict(torch.load(f"../experiments/models/Disc/{VERSION}/best_loss_disc_v3_double_Adam_0.5_0.1.pth", map_location=DEVICE)['model_state_dict'])
    disc_model.eval()

    pool_gmm = collect_pmf_samples(test_loader, disc_model, model_type="GMM")
    pool_ib = collect_pmf_samples(test_loader, disc_model, model_type="IB")

    final_data = {
        'GMM': filter_balanced_samples(pool_gmm, n_samples=16, param_key='scale'),
        'IB': filter_balanced_samples(pool_ib, n_samples=16, param_key='beta')
    }

    plot_comprehensive_interventions(final_data)