import os
import sys
import torch
import importlib
import math
import itertools
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from scipy import stats
import contextlib
from torch.amp import autocast
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join('..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../priors')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from utils import set_seed
from GaussianPrior import GaussianPrior

parser = argparse.ArgumentParser(description='Evaluate CC and Style on Colored MNIST (Gaussian + Post-Hoc).')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for dataloader')
args = parser.parse_args()

VERSION = '2_attr'
ARR_NUM_CLASSES = [10, 10]
NUM_ATTR = len(ARR_NUM_CLASSES)

DISC_CONFIG = {
    "MODEL": "disc_v3_double", 
    "PATH": f"../experiments/models/Disc/{VERSION}/best_loss_disc_v3_double_Adam_0.5_0.1.pth",
    "DROPOUT": 0.1
}

GEN_CONFIG = {
    "MODEL": "hybrid_v3_1x1_double",
    "DROPOUT": 0.1,
    "TOTAL_DIM": 4704                                 
}

MASTER_SEED = 42
rng = np.random.default_rng(MASTER_SEED)
SEEDS = ["original"] + rng.integers(0, 1000000, size=4).tolist()

MODELS_TO_LOAD = {}
for seed in SEEDS:
    if seed == "original":
        path = f"../experiments/models/Gaussian/{VERSION}/best_loss_hybrid_v3_1x1_double_GaussianPrior_Adam_0.5_0.1.pth"
        MODELS_TO_LOAD["Original"] = path
    else:
        path = f"../experiments_seed/models/Gaussian/{VERSION}/best_loss_hybrid_v3_1x1_double_GaussianPrior_Adam_0.5_0.1_{seed}.pth"
        MODELS_TO_LOAD[f"Seed_{seed}"] = path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMBDAS = [1.0]

BASE_COLORS = torch.tensor([
    [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]
], dtype=torch.float32)

def get_confidence_interval(data_array, confidence=0.95):
    """Calculates mean and error margin for CI."""
    data_array = np.array(data_array, dtype=float)
    if np.isnan(data_array).all():
        return np.nan, 0.0
    if len(data_array) < 2:
        return np.nanmean(data_array), 0.0
    
    mean = np.nanmean(data_array)
    std_err = stats.sem(data_array, nan_policy='omit')
    if isinstance(std_err, np.ma.core.MaskedConstant) or np.isnan(std_err):
        return mean, 0.0
        
    h = std_err * stats.t.ppf((1 + confidence) / 2., len(data_array) - 1)
    return mean, h

def get_dominant_colors(imgs, device):
    imgs_shifted = imgs + 0.5 
    
    left_half = imgs_shifted[:, :, :, :28]
    right_half = imgs_shifted[:, :, :, 28:]
    
    mask_left = (left_half.max(dim=1, keepdim=True)[0] > 0.2).float()
    mask_right = (right_half.max(dim=1, keepdim=True)[0] > 0.2).float()
    
    mean_left = (left_half * mask_left).sum(dim=(2, 3)) / (mask_left.sum(dim=(2, 3)) + 1e-5)
    mean_right = (right_half * mask_right).sum(dim=(2, 3)) / (mask_right.sum(dim=(2, 3)) + 1e-5)
    
    base_colors = BASE_COLORS.to(device)
    
    color_left = torch.argmin(torch.cdist(mean_left, base_colors), dim=1)
    color_right = torch.argmin(torch.cdist(mean_right, base_colors), dim=1)
    
    return color_left, color_right

def compute_centroids(model, loader, arr_num_classes, device):
    """Computes Latent Centroids for both Combinatorial and Independent approaches."""
    model.eval()
    num_attr = len(arr_num_classes)
    
    all_z = []
    all_y = []
    
    with torch.no_grad():
        for batch_X, batch_y in tqdm(loader, desc="Extracting Latents", leave=False):
            batch_X = batch_X.to(device)
            batch_X = (batch_X * 255.0 + torch.rand_like(batch_X)) / 256.0 - 0.5
            
            z, _ = model(batch_X)
            all_z.append(z.view(z.size(0), -1).cpu())
            all_y.append(batch_y.cpu())
            
    all_z = torch.cat(all_z, dim=0).float()
    all_y = torch.cat(all_y, dim=0)
    latent_dim = all_z.size(1)
    
    indep_centroids = []
    for k in range(num_attr):
        k_centroids = torch.zeros((arr_num_classes[k], latent_dim))
        for c in range(arr_num_classes[k]):
            mask = (all_y[:, k] == c)
            if mask.sum() > 0:
                k_centroids[c] = all_z[mask].mean(dim=0)
            else:
                k_centroids[c] = torch.full((latent_dim,), float('inf'))
        indep_centroids.append(k_centroids.to(device))
        
    ranges = [range(c) for c in arr_num_classes]
    all_combinations = list(itertools.product(*ranges))
    
    comb_centroids = []
    valid_combs = []
    
    for comb in all_combinations:
        comb_tensor = torch.tensor(comb)
        mask = (all_y == comb_tensor).all(dim=1)
        if mask.sum() > 0:
            comb_centroids.append(all_z[mask].mean(dim=0))
        else:
            comb_centroids.append(torch.full((latent_dim,), float('inf')))
        valid_combs.append(comb)
            
    comb_centroids = torch.stack(comb_centroids).to(device)
    valid_combs = torch.tensor(valid_combs, device=device)
    
    return comb_centroids, valid_combs, indep_centroids

def evaluate_single_model(gen, disc, prior, loader, device, mode, model_name):
    gen.eval()
    disc.eval()
    prior.eval()
    
    stats_tracker = {
        lam: {
            "attr0": {"cc_tgt": 0, "cc_non_tgt": 0, "cc_overall": 0, "style_tgt": 0, "style_non_tgt": 0, "style_overall": 0, "total": 0},
            "attr1": {"cc_tgt": 0, "cc_non_tgt": 0, "cc_overall": 0, "style_tgt": 0, "style_non_tgt": 0, "style_overall": 0, "total": 0},
            "both":  {"cc_tgt": 0, "cc_non_tgt": 0, "cc_overall": 0, "style_tgt": 0, "style_non_tgt": 0, "style_overall": 0, "total": 0}
        } for lam in LAMBDAS
    }

    with torch.no_grad():
        with autocast(device_type='cuda', dtype=torch.float16) if torch.cuda.is_available() else contextlib.nullcontext():
            for batch_X, batch_y in tqdm(loader, desc=f"Eval {model_name} ({mode})", leave=False):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                B = batch_X.size(0)
                
                batch_X = (batch_X * 255.0 + torch.rand_like(batch_X)) / 256.0
                batch_X_centered = batch_X - 0.5
                
                orig_color_left, orig_color_right = get_dominant_colors(batch_X_centered, device)
                
                logits_pre_tuple = disc(batch_X_centered)
                if not isinstance(logits_pre_tuple, (list, tuple)): logits_pre_tuple = (logits_pre_tuple,)
                preds_pre = torch.stack([torch.argmax(logits, dim=1) for logits in logits_pre_tuple], dim=1)
                
                z, _ = gen(batch_X_centered)
                z_flat = z.view(B, -1)
                
                for offset0 in range(10):
                    for offset1 in range(10):
                        if offset0 == 0 and offset1 == 0: continue
                        
                        t0 = (preds_pre[:, 0] + offset0) % 10
                        t1 = (preds_pre[:, 1] + offset1) % 10
                        
                        if offset0 != 0 and offset1 == 0: int_type = "attr0"
                        elif offset0 == 0 and offset1 != 0: int_type = "attr1"
                        else: int_type = "both"
                        
                        if mode == 'independent':
                            mu_orig_0 = prior.independent_means[0][preds_pre[:, 0]].to(device)
                            mu_tgt_0  = prior.independent_means[0][t0].to(device)
                            delta_0 = mu_tgt_0 - mu_orig_0
                            
                            mu_orig_1 = prior.independent_means[1][preds_pre[:, 1]].to(device)
                            mu_tgt_1  = prior.independent_means[1][t1].to(device)
                            delta_1 = mu_tgt_1 - mu_orig_1
                            
                            if int_type == "attr0": delta = delta_0
                            elif int_type == "attr1": delta = delta_1
                            else: delta = delta_0 + delta_1
                            
                        elif mode == 'combinatorial':
                            flat_orig = preds_pre[:, 0] * 10 + preds_pre[:, 1]
                            flat_tgt  = t0 * 10 + t1
                            
                            mu_orig_comb = prior.combinatorial_means[flat_orig].to(device)
                            mu_tgt_comb  = prior.combinatorial_means[flat_tgt].to(device)
                            
                            delta = mu_tgt_comb - mu_orig_comb
                        
                        z_interv_list = []
                        for lam in LAMBDAS:
                            z_interv = z_flat + lam * delta
                            z_interv_list.append(z_interv)
                            
                        z_interv_batched = torch.cat(z_interv_list, dim=0)
                        
                        z_interv_structural = z_interv_batched.view(len(LAMBDAS) * B, 12, 14, 28)
                        img_interv_batched = gen.inverse(z_interv_structural)
                        
                        logits_post_tuple = disc(img_interv_batched)
                        if not isinstance(logits_post_tuple, (list, tuple)): logits_post_tuple = (logits_post_tuple,)
                        preds_post_batched = torch.stack([torch.argmax(logits, dim=1) for logits in logits_post_tuple], dim=1)
                        
                        interv_color_left, interv_color_right = get_dominant_colors(img_interv_batched, device)
                        
                        for i, lam in enumerate(LAMBDAS):
                            preds_post = preds_post_batched[i * B : (i + 1) * B]
                            col_left = interv_color_left[i * B : (i + 1) * B]
                            col_right = interv_color_right[i * B : (i + 1) * B]
                            
                            correct_attr0 = (preds_post[:, 0] == t0)
                            correct_attr1 = (preds_post[:, 1] == t1)
                            
                            style_match_0 = (col_left == orig_color_left)
                            style_match_1 = (col_right == orig_color_right)

                            cc_overall_match = (correct_attr0 & correct_attr1)
                            
                            stats_tracker[lam][int_type]["style_overall"] += (style_match_0 & style_match_1).sum().item()
                            stats_tracker[lam][int_type]["cc_overall"] += cc_overall_match.sum().item()

                            if int_type == "attr0":
                                stats_tracker[lam][int_type]["cc_tgt"] += correct_attr0.sum().item()
                                stats_tracker[lam][int_type]["cc_non_tgt"] += correct_attr1.sum().item()
                                stats_tracker[lam][int_type]["style_tgt"] += style_match_0.sum().item()
                                stats_tracker[lam][int_type]["style_non_tgt"] += style_match_1.sum().item()
                            elif int_type == "attr1":
                                stats_tracker[lam][int_type]["cc_tgt"] += correct_attr1.sum().item()
                                stats_tracker[lam][int_type]["cc_non_tgt"] += correct_attr0.sum().item()
                                stats_tracker[lam][int_type]["style_tgt"] += style_match_1.sum().item()
                                stats_tracker[lam][int_type]["style_non_tgt"] += style_match_0.sum().item()
                            elif int_type == "both":
                                stats_tracker[lam][int_type]["cc_tgt"] += cc_overall_match.sum().item()                             
                                stats_tracker[lam][int_type]["style_tgt"] += (style_match_0 & style_match_1).sum().item()
                                
                            stats_tracker[lam][int_type]["total"] += B

    results = []
    for lam in LAMBDAS:
        for int_type in ["attr0", "attr1", "both"]:
            s = stats_tracker[lam][int_type]
            t_total = s["total"]
            num_interventions = 1 if int_type in ["attr0", "attr1"] else 2
            
            results.append({
                'Model': model_name,
                'lambda': lam,
                'intervention_group': int_type,
                'concepts_intervened': num_interventions,
                'cc_target': s["cc_tgt"] / t_total if t_total > 0 else np.nan,
                'cc_non_target': s["cc_non_tgt"] / t_total if t_total > 0 and int_type != "both" else np.nan,
                'cc_overall': s["cc_overall"] / t_total if t_total > 0 else np.nan,
                'style_target': s["style_tgt"] / t_total if t_total > 0 else np.nan,
                'style_non_target': s["style_non_tgt"] / t_total if t_total > 0 and int_type != "both" else np.nan,
                'style_overall': s["style_overall"] / t_total if t_total > 0 else np.nan,
                'total_samples': t_total
            })
            
        tot_1_cc_tgt = stats_tracker[lam]["attr0"]["cc_tgt"] + stats_tracker[lam]["attr1"]["cc_tgt"]
        tot_1_cc_ntgt = stats_tracker[lam]["attr0"]["cc_non_tgt"] + stats_tracker[lam]["attr1"]["cc_non_tgt"]
        tot_1_cc_ovr = stats_tracker[lam]["attr0"]["cc_overall"] + stats_tracker[lam]["attr1"]["cc_overall"]
        tot_1_st_tgt = stats_tracker[lam]["attr0"]["style_tgt"] + stats_tracker[lam]["attr1"]["style_tgt"]
        tot_1_st_ntgt = stats_tracker[lam]["attr0"]["style_non_tgt"] + stats_tracker[lam]["attr1"]["style_non_tgt"]
        tot_1_st_ovr = stats_tracker[lam]["attr0"]["style_overall"] + stats_tracker[lam]["attr1"]["style_overall"]
        tot_1_total = stats_tracker[lam]["attr0"]["total"] + stats_tracker[lam]["attr1"]["total"]
        
        results.append({
            'Model': model_name,
            'lambda': lam,
            'intervention_group': 'AGGREGATE_1_INTERVENTION',
            'concepts_intervened': 1,
            'cc_target': tot_1_cc_tgt / tot_1_total if tot_1_total > 0 else np.nan,
            'cc_non_target': tot_1_cc_ntgt / tot_1_total if tot_1_total > 0 else np.nan,
            'cc_overall': tot_1_cc_ovr / tot_1_total if tot_1_total > 0 else np.nan,
            'style_target': tot_1_st_tgt / tot_1_total if tot_1_total > 0 else np.nan,
            'style_non_target': tot_1_st_ntgt / tot_1_total if tot_1_total > 0 else np.nan,
            'style_overall': tot_1_st_ovr / tot_1_total if tot_1_total > 0 else np.nan,
            'total_samples': tot_1_total
        })

        tot_all_cc_tgt = tot_1_cc_tgt + stats_tracker[lam]["both"]["cc_tgt"]
        tot_all_cc_ovr = tot_1_cc_ovr + stats_tracker[lam]["both"]["cc_overall"]
        tot_all_st_tgt = tot_1_st_tgt + stats_tracker[lam]["both"]["style_tgt"]
        tot_all_st_ovr = tot_1_st_ovr + stats_tracker[lam]["both"]["style_overall"]
        tot_all_total = tot_1_total + stats_tracker[lam]["both"]["total"]
        
        results.append({
            'Model': model_name,
            'lambda': lam,
            'intervention_group': 'ALL',
            'concepts_intervened': 'ALL',
            'cc_target': tot_all_cc_tgt / tot_all_total if tot_all_total > 0 else np.nan,
            'cc_non_target': tot_1_cc_ntgt / tot_1_total if tot_1_total > 0 else np.nan,                    
            'cc_overall': tot_all_cc_ovr / tot_all_total if tot_all_total > 0 else np.nan,
            'style_target': tot_all_st_tgt / tot_all_total if tot_all_total > 0 else np.nan,
            'style_non_target': tot_1_st_ntgt / tot_1_total if tot_1_total > 0 else np.nan,
            'style_overall': tot_all_st_ovr / tot_all_total if tot_all_total > 0 else np.nan,
            'total_samples': tot_all_total
        })

    return results


if __name__ == "__main__":
    set_seed(42)
    
    print(f"Loading Calibration Dataset...")
    data = np.load('../data/easy_colored_double_mnist.npz')
    X_train, y_train = data['X_train'], data['y_train']
    y_train = y_train[:, 0:2]                      
    
    X_train_tensor = torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    
    X_test, y_test = data['X_test'], data['y_test']
    y_test = y_test[:, 0:2]
    X_test_tensor = torch.tensor(X_test.transpose(0, 3, 1, 2), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("\nLoading Discriminator...")
    disc_module = importlib.import_module(DISC_CONFIG['MODEL'])
    PseudoResNet = getattr(disc_module, 'PseudoResNet')
    disc_model = PseudoResNet(arr_num_classes=ARR_NUM_CLASSES, in_channels=3, dropout_p=DISC_CONFIG['DROPOUT'], device=DEVICE).to(DEVICE)
    disc_model.load_state_dict(torch.load(DISC_CONFIG['PATH'], map_location=DEVICE)['model_state_dict'])
    
    print("Loading Gaussian Generator Architecture...")
    gen_module = importlib.import_module(GEN_CONFIG['MODEL'])
    GeneralFlow = getattr(gen_module, 'GeneralFlow')
    gen = GeneralFlow(dropout_p=GEN_CONFIG['DROPOUT']).to(DEVICE)
    prior = GaussianPrior(device=DEVICE, num_attr=NUM_ATTR).to(DEVICE)

    if len(test_loader) > 0:
        all_results = {'independent': [], 'combinatorial': []}
        
        for model_name, model_path in MODELS_TO_LOAD.items():
            if not os.path.exists(model_path):
                print(f"Skipping {model_name}: Path not found ({model_path})")
                continue
                
            print(f"\nLoading weights and computing centroids for {model_name}...")
            checkpoint = torch.load(model_path, map_location=DEVICE)
            gen.load_state_dict(checkpoint['model_state_dict'])
            prior.load_state_dict(checkpoint['prior_state_dict'])
            
            comb_centroids, _, indep_centroids = compute_centroids(gen, train_loader, ARR_NUM_CLASSES, DEVICE)
            prior.independent_means = indep_centroids
            prior.combinatorial_means = comb_centroids
            
            for mode in ['independent', 'combinatorial']:
                res = evaluate_single_model(gen, disc_model, prior, test_loader, DEVICE, mode, model_name)
                all_results[mode].extend(res)

        for mode in ['independent', 'combinatorial']:
            if len(all_results[mode]) == 0:
                continue

            df_flat = pd.DataFrame(all_results[mode])
            
            metrics_to_agg = ['cc_target', 'cc_non_target', 'cc_overall', 'style_target', 'style_non_target', 'style_overall']
                              
            pivot_df = df_flat.pivot(index=['lambda', 'intervention_group', 'concepts_intervened', 'total_samples'], 
                                     columns='Model', 
                                     values=metrics_to_agg)
            
            pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]
            pivot_df = pivot_df.reset_index()

            for m in metrics_to_agg:
                model_cols = [col for col in pivot_df.columns if col.startswith(f"{m}_")]
                
                def apply_ci(row):
                    data = row[model_cols].values.astype(float)
                    mean, ci = get_confidence_interval(data, confidence=0.95)
                    return pd.Series([mean, ci])
                    
                pivot_df[[f"{m}_mean", f"{m}_ci_95"]] = pivot_df.apply(apply_ci, axis=1)

            results_csv = f"csv/Gaussian_Colored_{VERSION}/{mode}/cyc_cons_Gaussian_aggregate.csv"
            os.makedirs(os.path.dirname(results_csv), exist_ok=True)
            pivot_df.to_csv(results_csv, index=False)
            
            print(f"\nEvaluation complete for {mode}. Aggregated metrics saved to '{results_csv}'.")