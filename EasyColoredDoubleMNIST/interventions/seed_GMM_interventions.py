import os
import sys
import torch
import random
import importlib
import math
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from scipy import stats

sys.path.append(os.path.abspath(os.path.join('..', '..')))
sys.path.append(os.path.join(os.path.dirname(__file__), '../priors'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))
from utils import set_seed
from CheckerboardGMM import CheckerboardGMM

parser = argparse.ArgumentParser(description='Evaluate Cyclic Consistency and Style on Colored MNIST.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for the GMM means')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for dataloader')
args = parser.parse_args()

SCALE = args.scale

MODEL_CONFIG = {
    "MODEL": "hybrid_v3_1x1_double", 
    "PRIOR": "CheckerboardGMM",
    "DROPOUT": 0.1
}

MASTER_SEED = 42
rng = np.random.default_rng(MASTER_SEED)
SEEDS = ["original"] + rng.integers(0, 1000000, size=4).tolist()

MODELS_TO_LOAD = {}
for seed in SEEDS:
    if seed == "original":
        path = f"../experiments/models/GMM/2_attr/best_loss_{SCALE}_hybrid_v3_1x1_double_CheckerboardGMM_Adam_0.5_0.1.pth"
        MODELS_TO_LOAD["Original"] = path
    else:
        path = f"../experiments_seed/models/GMM/2_attr/best_loss_{SCALE}_hybrid_v3_1x1_double_CheckerboardGMM_Adam_0.5_0.1_{seed}.pth"
        MODELS_TO_LOAD[f"Seed_{seed}"] = path

DISC_CONFIG = {
    "MODEL": "disc_v3_double",
    "PATH": f"../experiments/models/Disc/2_attr/best_loss_disc_v3_double_Adam_0.5_0.1.pth",
    "DROPOUT": 0.1
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_CSV = f"csv/GMM_Scale_{SCALE}/cyc_cons_GMM_aggregate.csv"

LAMBDAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, "tilde"]

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
if os.path.exists(RESULTS_CSV):
    os.remove(RESULTS_CSV)

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

def parse_predictions(preds_raw):
    if isinstance(preds_raw, list):
        return torch.stack(preds_raw, dim=1)
    elif isinstance(preds_raw, tuple):
        return torch.stack(preds_raw[0], dim=1) if isinstance(preds_raw[0], list) else preds_raw[0]
    return preds_raw

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

def compute_tilde_lambda(z, mu_orig, mu_target, all_means, t_target):
    v = z - mu_orig
    diff = all_means.unsqueeze(0) - mu_target.unsqueeze(1)
    dot_product = (diff * v.unsqueeze(1)).sum(dim=-1)
    dist_sq = (diff ** 2).sum(dim=-1)
    
    eps = 1e-7
    lam_bounds = torch.where(dot_product > eps, dist_sq / (2 * dot_product + eps), torch.tensor(float('inf'), device=z.device))
    
    t_mask = torch.arange(10, device=z.device).unsqueeze(0) == t_target.unsqueeze(1)
    lam_bounds.masked_fill_(t_mask, float('inf'))
    
    tilde_lam = lam_bounds.min(dim=1)[0]
    return tilde_lam

def evaluate_single_model(model, prior, disc_model, loader, device, model_name):
    model.eval()
    prior.eval()
    disc_model.eval() 
    
    stats_tracker = { lam: { attr: {"cc_tgt": 0, "cc_non_tgt": 0, "style_tgt": 0, "style_non_tgt": 0, 
                                    "style_overall": 0, "disc_left": 0, "disc_right": 0, "disc_overall": 0, "total": 0} 
                             for attr in ["attr0", "attr1", "both"] } for lam in LAMBDAS }

    with torch.no_grad():
        for batch_X, batch_y in tqdm(loader, desc=f"Evaluating {model_name}", leave=False):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            B = batch_X.size(0)
            
            batch_X = (batch_X * 255.0 + torch.rand_like(batch_X)) / 256.0
            batch_X_centered = batch_X - 0.5
            
            orig_color_left, orig_color_right = get_dominant_colors(batch_X_centered, device)
            z, _ = model(batch_X_centered)
            z_flat = z.view(B, -1)
            
            preds_pre_raw, _ = prior.classify(z_flat)
            preds_pre = parse_predictions(preds_pre_raw)
            
            z_parts = prior.get_parts(z_flat)
            left_z, right_z = z_parts[0], z_parts[1]
            
            mu_left_orig = prior.means[0][preds_pre[:, 0]].to(device)
            mu_right_orig = prior.means[1][preds_pre[:, 1]].to(device)
            
            for offset0 in range(10):
                for offset1 in range(10):
                    if offset0 == 0 and offset1 == 0: continue
                    
                    t0 = (preds_pre[:, 0] + offset0) % 10
                    t1 = (preds_pre[:, 1] + offset1) % 10
                    
                    if offset0 != 0 and offset1 == 0: int_type = "attr0"
                    elif offset0 == 0 and offset1 != 0: int_type = "attr1"
                    else: int_type = "both"
                        
                    mu_left_target = prior.means[0][t0].to(device)
                    mu_right_target = prior.means[1][t1].to(device)

                    tilde_lam_0 = compute_tilde_lambda(left_z, mu_left_orig, mu_left_target, prior.means[0], t0)
                    tilde_lam_1 = compute_tilde_lambda(right_z, mu_right_orig, mu_right_target, prior.means[1], t1)
                    
                    if int_type == "attr0": tilde_lam = tilde_lam_0
                    elif int_type == "attr1": tilde_lam = tilde_lam_1
                    else: tilde_lam = torch.min(tilde_lam_0, tilde_lam_1)
                        
                    tilde_lam = torch.clamp(tilde_lam * 0.999, min=0.0, max=1.0)

                    z_interv_list = []
                    for lam in LAMBDAS:
                        if lam == "tilde":
                            left_z_shifted = mu_left_target + tilde_lam.unsqueeze(1) * (left_z - mu_left_orig)
                            right_z_shifted = mu_right_target + tilde_lam.unsqueeze(1) * (right_z - mu_right_orig)
                        else:
                            left_z_shifted = mu_left_target + lam * (left_z - mu_left_orig)
                            right_z_shifted = mu_right_target + lam * (right_z - mu_right_orig)
                        
                        if int_type == "attr0": z_interv = prior.get_full_latent([left_z_shifted, right_z])
                        elif int_type == "attr1": z_interv = prior.get_full_latent([left_z, right_z_shifted])
                        elif int_type == "both": z_interv = prior.get_full_latent([left_z_shifted, right_z_shifted])
                            
                        z_interv_list.append(z_interv)

                    z_interv_batched = torch.cat(z_interv_list, dim=0)
                    z_interv_structural = z_interv_batched.view(len(LAMBDAS) * B, 12, 14, 28)
                    img_interv_batched = model.inverse(z_interv_structural)

                    preds_post_raw, _ = prior.classify(z_interv_structural.view(len(LAMBDAS) * B, -1))
                    preds_post_batched = parse_predictions(preds_post_raw)
                    interv_color_left, interv_color_right = get_dominant_colors(img_interv_batched, device)
                    
                    disc_logits = disc_model(img_interv_batched)
                    disc_preds = torch.stack([logit.argmax(dim=1) for logit in disc_logits], dim=1)

                    for i, lam in enumerate(LAMBDAS):
                        preds_post = preds_post_batched[i * B : (i + 1) * B]
                        col_left = interv_color_left[i * B : (i + 1) * B]
                        col_right = interv_color_right[i * B : (i + 1) * B]
                        disc_p = disc_preds[i * B : (i + 1) * B] 
                        
                        correct_attr0 = (preds_post[:, 0] == t0)
                        correct_attr1 = (preds_post[:, 1] == t1)
                        
                        style_match_0 = (col_left == orig_color_left)
                        style_match_1 = (col_right == orig_color_right)
                        
                        disc_correct_left = (disc_p[:, 0] == t0)
                        disc_correct_right = (disc_p[:, 1] == t1)
                        disc_correct_overall = disc_correct_left & disc_correct_right
                        
                        stats_tracker[lam][int_type]["disc_left"] += disc_correct_left.sum().item()
                        stats_tracker[lam][int_type]["disc_right"] += disc_correct_right.sum().item()
                        stats_tracker[lam][int_type]["disc_overall"] += disc_correct_overall.sum().item()
                        stats_tracker[lam][int_type]["style_overall"] += (style_match_0 & style_match_1).sum().item()
                        
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
                            stats_tracker[lam][int_type]["cc_tgt"] += (correct_attr0 & correct_attr1).sum().item()
                            stats_tracker[lam][int_type]["style_tgt"] += (style_match_0 & style_match_1).sum().item()
                            
                        stats_tracker[lam][int_type]["total"] += B

    results = []
    for lam in LAMBDAS:
        for int_type in ["attr0", "attr1", "both"]:
            s = stats_tracker[lam][int_type]
            t_total = s["total"]
            
            results.append({
                'Model': model_name,
                'lambda': lam, 'intervention_group': int_type, 'concepts_intervened': 1 if int_type in ["attr0", "attr1"] else 2,
                'cc_target': s["cc_tgt"] / t_total if t_total > 0 else np.nan,
                'cc_non_target': s["cc_non_tgt"] / t_total if t_total > 0 and int_type != "both" else np.nan,
                'style_target': s["style_tgt"] / t_total if t_total > 0 else np.nan,
                'style_non_target': s["style_non_tgt"] / t_total if t_total > 0 and int_type != "both" else np.nan,
                'style_overall': s["style_overall"] / t_total if t_total > 0 else np.nan,
                'disc_left_acc': s["disc_left"] / t_total if t_total > 0 else np.nan,
                'disc_right_acc': s["disc_right"] / t_total if t_total > 0 else np.nan,
                'disc_overall_acc': s["disc_overall"] / t_total if t_total > 0 else np.nan,
                'total_samples': t_total
            })
            
        tot_1_cc_tgt = stats_tracker[lam]["attr0"]["cc_tgt"] + stats_tracker[lam]["attr1"]["cc_tgt"]
        tot_1_cc_ntgt = stats_tracker[lam]["attr0"]["cc_non_tgt"] + stats_tracker[lam]["attr1"]["cc_non_tgt"]
        tot_1_st_tgt = stats_tracker[lam]["attr0"]["style_tgt"] + stats_tracker[lam]["attr1"]["style_tgt"]
        tot_1_st_ntgt = stats_tracker[lam]["attr0"]["style_non_tgt"] + stats_tracker[lam]["attr1"]["style_non_tgt"]
        tot_1_st_ovr = stats_tracker[lam]["attr0"]["style_overall"] + stats_tracker[lam]["attr1"]["style_overall"]
        tot_1_disc_l = stats_tracker[lam]["attr0"]["disc_left"] + stats_tracker[lam]["attr1"]["disc_left"]
        tot_1_disc_r = stats_tracker[lam]["attr0"]["disc_right"] + stats_tracker[lam]["attr1"]["disc_right"]
        tot_1_disc_o = stats_tracker[lam]["attr0"]["disc_overall"] + stats_tracker[lam]["attr1"]["disc_overall"]
        tot_1_total = stats_tracker[lam]["attr0"]["total"] + stats_tracker[lam]["attr1"]["total"]
        
        results.append({
            'Model': model_name,
            'lambda': lam, 'intervention_group': 'AGGREGATE_1_INTERVENTION', 'concepts_intervened': 1,
            'cc_target': tot_1_cc_tgt / tot_1_total if tot_1_total > 0 else np.nan,
            'cc_non_target': tot_1_cc_ntgt / tot_1_total if tot_1_total > 0 else np.nan,
            'style_target': tot_1_st_tgt / tot_1_total if tot_1_total > 0 else np.nan,
            'style_non_target': tot_1_st_ntgt / tot_1_total if tot_1_total > 0 else np.nan,
            'style_overall': tot_1_st_ovr / tot_1_total if tot_1_total > 0 else np.nan,
            'disc_left_acc': tot_1_disc_l / tot_1_total if tot_1_total > 0 else np.nan,
            'disc_right_acc': tot_1_disc_r / tot_1_total if tot_1_total > 0 else np.nan,
            'disc_overall_acc': tot_1_disc_o / tot_1_total if tot_1_total > 0 else np.nan,
            'total_samples': tot_1_total
        })

        tot_all_cc_tgt = tot_1_cc_tgt + stats_tracker[lam]["both"]["cc_tgt"]
        tot_all_st_tgt = tot_1_st_tgt + stats_tracker[lam]["both"]["style_tgt"]
        tot_all_st_ovr = tot_1_st_ovr + stats_tracker[lam]["both"]["style_overall"]
        tot_all_disc_l = tot_1_disc_l + stats_tracker[lam]["both"]["disc_left"]
        tot_all_disc_r = tot_1_disc_r + stats_tracker[lam]["both"]["disc_right"]
        tot_all_disc_o = tot_1_disc_o + stats_tracker[lam]["both"]["disc_overall"]
        tot_all_total = tot_1_total + stats_tracker[lam]["both"]["total"]
        
        results.append({
            'Model': model_name,
            'lambda': lam, 'intervention_group': 'ALL', 'concepts_intervened': 'ALL',
            'cc_target': tot_all_cc_tgt / tot_all_total if tot_all_total > 0 else np.nan,
            'cc_non_target': tot_1_cc_ntgt / tot_1_total if tot_1_total > 0 else np.nan,                         
            'style_target': tot_all_st_tgt / tot_all_total if tot_all_total > 0 else np.nan,
            'style_non_target': tot_1_st_ntgt / tot_1_total if tot_1_total > 0 else np.nan,
            'style_overall': tot_all_st_ovr / tot_all_total if tot_all_total > 0 else np.nan,
            'disc_left_acc': tot_all_disc_l / tot_all_total if tot_all_total > 0 else np.nan,
            'disc_right_acc': tot_all_disc_r / tot_all_total if tot_all_total > 0 else np.nan,
            'disc_overall_acc': tot_all_disc_o / tot_all_total if tot_all_total > 0 else np.nan,
            'total_samples': tot_all_total
        })

    return results


if __name__ == "__main__":
    set_seed(42)
    
    print("Loading Base Architecture...")
    model_module = importlib.import_module(MODEL_CONFIG['MODEL'])
    GeneralFlow = getattr(model_module, 'GeneralFlow')
    
    prior_class = globals()[MODEL_CONFIG['PRIOR']]
    prior = prior_class(total_dim=4704, arr_num_classes=[10, 10], device=DEVICE, scale=SCALE, fixed_means=True)
    prior.num_classes = 20
    model = GeneralFlow(dropout_p=MODEL_CONFIG['DROPOUT']).to(DEVICE)
    prior.num_attr = 2
    prior.total_dim = 4704
    
    print("Loading Discriminative Classifier...")
    disc_module = importlib.import_module(DISC_CONFIG['MODEL'])
    PseudoResNet = getattr(disc_module, 'PseudoResNet')
    
    disc_model = PseudoResNet(arr_num_classes=[10, 10], in_channels=3, dropout_p=DISC_CONFIG['DROPOUT'], device=DEVICE).to(DEVICE)
    disc_checkpoint = torch.load(DISC_CONFIG['PATH'], map_location=DEVICE)
    disc_model.load_state_dict(disc_checkpoint['model_state_dict'])
    
    from torch.utils.data import TensorDataset, DataLoader
    
    print("Loading test dataset...")
    data = np.load('../data/easy_colored_double_mnist.npz')
    X_test, y_test = data['X_test'], data['y_test']
    
    y_test = y_test[:, 0:2]
    
    X_test_tensor = torch.tensor(X_test.transpose(0, 3, 1, 2), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    if len(test_loader) > 0:
        all_model_results = []
        
        for model_name, model_path in MODELS_TO_LOAD.items():
            if not os.path.exists(model_path):
                print(f"Skipping {model_name}: Path not found ({model_path})")
                continue
                
            print(f"\nLoading weights for {model_name}...")
            checkpoint = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            prior.load_state_dict(checkpoint['prior_state_dict'])
            prior.means = checkpoint['means']
            
            res = evaluate_single_model(model, prior, disc_model, test_loader, DEVICE, model_name)
            all_model_results.extend(res)
            
        if len(all_model_results) == 0:
            print("No models evaluated. Exiting.")
            sys.exit(0)

        df_flat = pd.DataFrame(all_model_results)
        
        metrics_to_agg = ['cc_target', 'cc_non_target', 'style_target', 'style_non_target', 
                          'style_overall', 'disc_left_acc', 'disc_right_acc', 'disc_overall_acc']
                          
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

        pivot_df.to_csv(RESULTS_CSV, index=False)
        print(f"\nEvaluation complete. Aggregated metrics saved to '{RESULTS_CSV}'.")
