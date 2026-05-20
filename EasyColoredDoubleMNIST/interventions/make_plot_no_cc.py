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

SCALES = [0.05,0.1]
BETAS = [0.01, 0.05, 0.1, 0.5, 1.0]
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

def collect_cnf_samples(loader, disc_model):
    print("Collecting CNF (Conditional) Samples (CC Failure, Style Success)...")
    pool = []
    
    gen_module = importlib.import_module("conditional_scale")
    gen = gen_module.GeneralFlow(num_classes=20, dropout_p=0.1, cond_dim=64).to(DEVICE)
    gen.load_state_dict(torch.load(f"../experiments/models/Conditional/{VERSION}/best_loss_conditional_scale_GaussianPrior_Adam_0.5_0.1.pth", map_location=DEVICE)['model_state_dict'])
    gen.eval()

    with torch.no_grad():
        for batch_X, _ in loader:
            batch_X = batch_X.to(DEVICE)
            batch_X = (batch_X * 255.0 + torch.rand_like(batch_X)) / 256.0
            batch_X_centered = batch_X - 0.5
            
            orig_c_left, orig_c_right = get_dominant_colors(batch_X_centered, DEVICE)
            preds_pre = torch.stack([torch.argmax(logits, dim=1) for logits in disc_model(batch_X_centered)], dim=1)
            y_cond_pre = torch.cat([F.one_hot(preds_pre[:, 0], 10), F.one_hot(preds_pre[:, 1], 10)], dim=1).float()
            z, _ = gen(batch_X_centered, y_cond_pre)
            
            for int_type in ['attr0', 'attr1', 'both']:
                t0, t1 = preds_pre[:, 0].clone(), preds_pre[:, 1].clone()
                if int_type in ['attr0', 'both']: t0 = (t0 + random.randint(1, 9)) % 10
                if int_type in ['attr1', 'both']: t1 = (t1 + random.randint(1, 9)) % 10
                
                y_cond_target = torch.cat([F.one_hot(t0, 10), F.one_hot(t1, 10)], dim=1).float()
                img_interv = gen.inverse(z, y_cond_target)
                
                preds_post = torch.stack([torch.argmax(logits, dim=1) for logits in disc_model(img_interv)], dim=1)
                int_c_left, int_c_right = get_dominant_colors(img_interv, DEVICE)
                
                cc_fail = (preds_post[:, 0] != t0) | (preds_post[:, 1] != t1)
                style_succ = (int_c_left == orig_c_left) & (int_c_right == orig_c_right)
                succ = cc_fail & style_succ
                
                for idx in torch.nonzero(succ, as_tuple=False).squeeze(-1):
                    pool.append({
                        'img_orig': batch_X[idx].cpu().numpy(),
                        'img_interv': img_interv[idx].cpu().numpy(),
                        'orig_classes': [preds_pre[idx, 0].item(), preds_pre[idx, 1].item()],
                        'tgt_classes': [t0[idx].item(), t1[idx].item()],
                        'pred_post': [preds_post[idx, 0].item(), preds_post[idx, 1].item()],
                        'int_type': int_type
                    })
            if len(pool) > 100: break
            
    del gen; torch.cuda.empty_cache()
    return pool

def collect_hce_samples(loader, disc_model, train_loader):
    print("Collecting HCE (Gaussian) Samples (CC Failure, Style Success)...")
    pool_c, pool_i = [], []
    
    gen_module = importlib.import_module("hybrid_v3_1x1_double")
    gen = gen_module.GeneralFlow(dropout_p=0.1).to(DEVICE)
    prior = GaussianPrior(device=DEVICE, num_attr=2).to(DEVICE)
    
    ckpt = torch.load(f"../experiments/models/Gaussian/{VERSION}/best_loss_hybrid_v3_1x1_double_GaussianPrior_Adam_0.5_0.1.pth", map_location=DEVICE)
    gen.load_state_dict(ckpt['model_state_dict'])
    prior.load_state_dict(ckpt['prior_state_dict'])
    gen.eval(); prior.eval()

    all_z, all_y = [], []
    with torch.no_grad():
        for batch_X, batch_y in train_loader:
            z, _ = gen(((batch_X.to(DEVICE) * 255.0 + torch.rand_like(batch_X.to(DEVICE))) / 256.0) - 0.5)
            all_z.append(z.view(z.size(0), -1).cpu()); all_y.append(batch_y.cpu())
            
    all_z = torch.cat(all_z, dim=0).float()
    all_y = torch.cat(all_y, dim=0)
    
    indep_centroids = [torch.zeros((10, all_z.size(1))).to(DEVICE) for _ in range(2)]
    for k in range(2):
        for c in range(10): indep_centroids[k][c] = all_z[all_y[:, k] == c].mean(dim=0)
    
    comb_centroids = torch.zeros((100, all_z.size(1))).to(DEVICE)
    for c0, c1 in itertools.product(range(10), range(10)):
        mask = (all_y[:, 0] == c0) & (all_y[:, 1] == c1)
        if mask.sum() > 0: comb_centroids[c0*10 + c1] = all_z[mask].mean(dim=0)

    with torch.no_grad():
        for batch_X, _ in loader:
            batch_X = batch_X.to(DEVICE)
            batch_X_centered = ((batch_X * 255.0 + torch.rand_like(batch_X)) / 256.0) - 0.5
            
            orig_c_left, orig_c_right = get_dominant_colors(batch_X_centered, DEVICE)
            preds_pre = torch.stack([torch.argmax(logits, dim=1) for logits in disc_model(batch_X_centered)], dim=1)
            z, _ = gen(batch_X_centered)
            z_flat = z.view(z.size(0), -1)
            
            for int_type in ['attr0', 'attr1', 'both']:
                t0, t1 = preds_pre[:, 0].clone(), preds_pre[:, 1].clone()
                if int_type in ['attr0', 'both']: t0 = (t0 + random.randint(1, 9)) % 10
                if int_type in ['attr1', 'both']: t1 = (t1 + random.randint(1, 9)) % 10
                
                delta_0 = indep_centroids[0][t0] - indep_centroids[0][preds_pre[:, 0]]
                delta_1 = indep_centroids[1][t1] - indep_centroids[1][preds_pre[:, 1]]
                delta_i = delta_0 if int_type == 'attr0' else (delta_1 if int_type == 'attr1' else delta_0 + delta_1)
                
                img_int_i = gen.inverse((z_flat + 1.0 * delta_i).view(z.size(0), 12, 14, 28))
                preds_post_i = torch.stack([torch.argmax(logits, dim=1) for logits in disc_model(img_int_i)], dim=1)
                c_left_i, c_right_i = get_dominant_colors(img_int_i, DEVICE)
                
                succ_i = ((preds_post_i[:, 0] != t0) | (preds_post_i[:, 1] != t1)) & ((c_left_i != orig_c_left) | (c_right_i != orig_c_right))
                for idx in torch.nonzero(succ_i, as_tuple=False).squeeze(-1):
                    pool_i.append({
                        'img_orig': batch_X[idx].cpu().numpy(), 'img_interv': img_int_i[idx].cpu().numpy(), 'pred_post': [preds_post_i[idx, 0].item(), preds_post_i[idx, 1].item()],
                        'orig_classes': [preds_pre[idx, 0].item(), preds_pre[idx, 1].item()], 'tgt_classes': [t0[idx].item(), t1[idx].item()], 'int_type': int_type
                    })

                flat_orig, flat_tgt = preds_pre[:, 0] * 10 + preds_pre[:, 1], t0 * 10 + t1
                img_int_c = gen.inverse((z_flat + 1.0 * (comb_centroids[flat_tgt] - comb_centroids[flat_orig])).view(z.size(0), 12, 14, 28))
                
                preds_post_c = torch.stack([torch.argmax(logits, dim=1) for logits in disc_model(img_int_c)], dim=1)
                c_left_c, c_right_c = get_dominant_colors(img_int_c, DEVICE)
                
                succ_c = ((preds_post_c[:, 0] != t0) | (preds_post_c[:, 1] != t1)) & ((c_left_c != orig_c_left) | (c_right_c != orig_c_right))
                for idx in torch.nonzero(succ_c, as_tuple=False).squeeze(-1):
                    pool_c.append({
                        'img_orig': batch_X[idx].cpu().numpy(), 'img_interv': img_int_c[idx].cpu().numpy(), 'pred_post': [preds_post_c[idx, 0].item(), preds_post_c[idx, 1].item()],
                        'orig_classes': [preds_pre[idx, 0].item(), preds_pre[idx, 1].item()], 'tgt_classes': [t0[idx].item(), t1[idx].item()], 'int_type': int_type
                    })
                    
            if len(pool_i) > 60 and len(pool_c) > 60: break
            
    del gen, prior; torch.cuda.empty_cache()
    return pool_c, pool_i

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
                        c_left, c_right = get_dominant_colors(img_interv, DEVICE)
                        
                        succ = ((preds_post_prior[:, 0] != t0) | (preds_post_prior[:, 1] != t1)) & ((c_left == orig_c_left) & (c_right == orig_c_right))
                        for idx in torch.nonzero(succ, as_tuple=False).squeeze(-1):
                            pool.append({
                                'img_orig': batch_X[idx].cpu().numpy(), 'img_interv': img_interv[idx].cpu().numpy(),
                                'orig_classes': [preds_pre[idx, 0].item(), preds_pre[idx, 1].item()],
                                'tgt_classes': [t0[idx].item(), t1[idx].item()],
                                'pred_post': [preds_post_prior[idx, 0].item(), preds_post_prior[idx, 1].item()],
                                'lambda': lam, 'scale' if model_type == "GMM" else 'beta': param, 'int_type': int_type
                            })
                if len(pool) >= 50: break
        del gen, prior; torch.cuda.empty_cache()
    return pool

def filter_balanced_samples(pool, n_samples, param_key=None):
    """Filters pool to ensure uniqueness, aims for 1/3 per type, but falls back to others if needed."""
    n_attr0 = n_samples // 3
    n_attr1 = n_samples // 3
    n_both = n_samples - n_attr0 - n_attr1
    
    targets = {'attr0': n_attr0, 'attr1': n_attr1, 'both': n_both}
    counts = {'attr0': 0, 'attr1': 0, 'both': 0}
    
    selected = []
    reserve = []
    seen_hashes = set()
    seen_params = set()
    
    random.shuffle(pool)
    
    for item in pool:
        itype = item['int_type']
        
        orig_hash = hash(item['img_orig'].tobytes())
        if orig_hash in seen_hashes: 
            continue
            
        if param_key:
            combo = (item['lambda'], item[param_key])
            if combo in seen_params: 
                continue

        if counts[itype] < targets[itype]:
            seen_hashes.add(orig_hash)
            if param_key: seen_params.add(combo)
            counts[itype] += 1
            selected.append(item)
            
            if len(selected) >= n_samples: 
                return selected
        else:
            reserve.append(item)
            
    for item in reserve:
        if len(selected) >= n_samples:
            break
            
        orig_hash = hash(item['img_orig'].tobytes())
        if orig_hash in seen_hashes:
            continue
            
        if param_key:
            combo = (item['lambda'], item[param_key])
            if combo in seen_params:
                continue
                
        seen_hashes.add(orig_hash)
        if param_key: seen_params.add(combo)
        selected.append(item)

    return selected

def plot_comprehensive_interventions(data, save_path="plots/comprehensive_interventions_no_cc.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = plt.figure(figsize=(19, 20), layout="constrained")
    main_subfigs = fig.subfigures(3, 1, height_ratios=[7.5, 4.13, 6.35])
    
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
            p_p = item['pred_post']
            class_str = f"[{o_c[0]}, {o_c[1]}] -> [{t_c[0]}, {t_c[1]}] Pred: [{p_p[0]}, {p_p[1]}]"
            
            if section_type == 'GMM':
                ax.set_title(class_str + "\n" + r"$s={} \mid \lambda={}$".format(item['scale'], item['lambda']), fontsize=20)
            elif section_type == 'IB':
                ax.set_title(class_str + "\n" + r"$\beta={} \mid \lambda={}$".format(item['beta'], item['lambda']), fontsize=20)
            else:
                ax.set_title(class_str, fontsize=20)

    main_subfigs[0].suptitle("PMF", fontsize=32, fontweight='bold')
    pmf_subfigs = main_subfigs[0].subfigures(2, 1, height_ratios=[1.5, 3.5])
    
    pmf_subfigs[0].suptitle("GMM", fontsize=28)
    populate_axes(pmf_subfigs[0].subplots(1, 4, gridspec_kw={'hspace': 0.0}), data['GMM'], 'GMM')
    
    pmf_subfigs[1].suptitle("IB", fontsize=28)
    populate_axes(pmf_subfigs[1].subplots(3, 5, gridspec_kw={'hspace': 0.0}), data['IB'], 'IB')

    main_subfigs[1].suptitle("CNF", fontsize=32, fontweight='bold')
    populate_axes(main_subfigs[1].subplots(3, 5, gridspec_kw={'hspace': 0.0}), data['CNF'], 'CNF')

    main_subfigs[2].suptitle("HCE", fontsize=32, fontweight='bold')
    hce_subfigs = main_subfigs[2].subfigures(2, 1)
    
    hce_subfigs[0].suptitle("Combinatorial", fontsize=28)
    populate_axes(hce_subfigs[0].subplots(2, 5, gridspec_kw={'hspace': 0.0}), data['HCE_C'], 'HCE_C')
    
    hce_subfigs[1].suptitle("Independent", fontsize=28)
    populate_axes(hce_subfigs[1].subplots(2, 5, gridspec_kw={'hspace': 0.0}), data['HCE_I'], 'HCE_I')

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

    pool_cnf = collect_cnf_samples(test_loader, disc_model)
    pool_hce_c, pool_hce_i = collect_hce_samples(test_loader, disc_model, train_loader)
    pool_gmm = collect_pmf_samples(test_loader, disc_model, model_type="GMM")
    pool_ib = collect_pmf_samples(test_loader, disc_model, model_type="IB")

    final_data = {
        'GMM': filter_balanced_samples(pool_gmm, n_samples=4, param_key='scale'),
        'IB': filter_balanced_samples(pool_ib, n_samples=15, param_key='beta'),
        'CNF': filter_balanced_samples(pool_cnf, n_samples=15),
        'HCE_C': filter_balanced_samples(pool_hce_c, n_samples=10),
        'HCE_I': filter_balanced_samples(pool_hce_i, n_samples=10)
    }

    plot_comprehensive_interventions(final_data)