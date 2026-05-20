import os
import sys
import torch
import random
import importlib
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse

sys.path.append(os.path.abspath(os.path.join('..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../priors')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from utils import set_seed
from CheckerboardGMM import CheckerboardGMM

parser = argparse.ArgumentParser(description='Train a flow-based model on MNIST.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for the GMM means (default: 1.0)')
args = parser.parse_args()

SCALE = args.scale
SEED = 42

ARR_NUM_CLASSES = [10, 10]
NUM_ATTR = len(ARR_NUM_CLASSES)
TOTAL_DIM = 1568                              

MODEL_CONFIG = {
    "MODEL": "hybrid_v3_1x1_double", 
    "PATH": f"../experiments/models/GMM/best_loss_{SCALE}_hybrid_v3_1x1_double_CheckerboardGMM_Adam_0.5_0.1.pth",
    "PRIOR": "CheckerboardGMM",
    "DROPOUT": 0.1,
    "TOTAL_DIM": TOTAL_DIM
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_CSV = f"csv/GMM_{SCALE}/class_cons_GMM.csv"
PLOT_DIR = f"plots/GMM_{SCALE}/"
SAMPLES_PER_CONCEPT = 1000 
TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8]

rng = np.random.default_rng(SEED)
SEEDS = ["original"] + rng.integers(0, 1000000, size=4).tolist()

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def get_confidence_interval(data_array, confidence=0.95):
    """Calculates mean and error margin for CI."""
    data_array = np.array(data_array)
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
    """Parses raw predictions from the GMM prior into a standardized tensor format."""
    if isinstance(preds_raw, list):
        return torch.stack(preds_raw, dim=1)
    elif isinstance(preds_raw, tuple):
        return torch.stack(preds_raw[0], dim=1) if isinstance(preds_raw[0], list) else preds_raw[0]
    return preds_raw

def plot_failure_grid(failures, temp, save_dir, max_plots=64):
    """Plots a grid of failures, showing desired vs. classified concepts for Double MNIST."""
    if not failures:
        print(f"    -> No failures to plot for Temperature {temp}.")
        return

    if len(failures) > max_plots:
        plot_samples = random.sample(failures, max_plots)
    else:
        plot_samples = failures

    cols = 8
    rows = math.ceil(len(plot_samples) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3))
    fig.suptitle(f"Cyclic Consistency Failures (T={temp})", fontsize=16, fontweight='bold', y=1.02)
    
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for k, ax in enumerate(axes):
        if k < len(plot_samples):
            f = plot_samples[k]
            
            ax.imshow(f['img'].squeeze(), cmap='gray')
            ax.set_title(f"Target: {f['desired']}\nPred: {f['pred']}", color='red', fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/consistency_failures_T_{temp}.png", bbox_inches='tight')
    plt.close(fig)

def evaluate_cyclic_consistency(model, prior, device, is_original=False):
    model.eval()
    prior.eval()
    
    temp_aggregates = {}
    
    all_combinations = list(itertools.product(*[range(n) for n in ARR_NUM_CLASSES]))
    total_combinations = len(all_combinations)
    
    for temp in TEMPERATURES:
        my_samples_per_concept = 1 if temp == 0.0 else SAMPLES_PER_CONCEPT
        print(f"  -> Evaluating Temperature: {temp} ({total_combinations} combinations)")
        
        failures_for_temp = []
        temp_consistencies = [] 
        
        with torch.no_grad():
            for combo_idx, combo in enumerate(all_combinations):
                
                z_parts = []
                for k, c in enumerate(combo):
                    mean_k = prior.means[k][c].to(device).unsqueeze(0).expand(my_samples_per_concept, -1)
                    z_parts.append(mean_k)
                
                z = prior.get_full_latent(z_parts)
                if temp > 0:
                    z = z + torch.randn_like(z) * temp
                    
                z_structural = z.view(my_samples_per_concept, 4, 14, 28)
                img_gen = model.inverse(z_structural)
                
                z_post, _ = model(img_gen)
                z_post_flat = z_post.view(my_samples_per_concept, -1)
                
                preds_post_raw, _ = prior.classify(z_post_flat)
                preds_post = parse_predictions(preds_post_raw)
                
                correct_all = torch.ones(my_samples_per_concept, dtype=torch.bool, device=device)
                for k, c in enumerate(combo):
                    correct_all &= (preds_post[:, k] == c)
                
                consistency_ratio = correct_all.float().mean().item()
                temp_consistencies.append(consistency_ratio)
                
                if is_original:
                    fail_mask = ~correct_all
                    if fail_mask.any():
                        fail_indices = torch.nonzero(fail_mask, as_tuple=False).squeeze(-1)
                        for idx in fail_indices[:2]: 
                            failures_for_temp.append({
                                'img': img_gen[idx].cpu().numpy(),
                                'desired': list(combo),
                                'pred': preds_post[idx].cpu().tolist()
                            })
        
        aggregate_ratio = float(np.mean(temp_consistencies))
        temp_aggregates[temp] = aggregate_ratio
        
            
    return temp_aggregates


if __name__ == "__main__":
    set_seed(SEED)
    
    print("Loading Generative Classifier Architecture...")
    model_module = importlib.import_module(MODEL_CONFIG['MODEL'])
    GeneralFlow = getattr(model_module, 'GeneralFlow')
    prior_class = globals()[MODEL_CONFIG['PRIOR']]
    
    all_results = []
    
    print(f"Configuration: DoubleMNIST | Total Attributes: {NUM_ATTR} | Total Classes: {ARR_NUM_CLASSES}")
    
    for seed in SEEDS:
        print(f"\n[{'='*40}]")
        print(f"Processing Model Seed: {seed}")
        print(f"[{'='*40}]")
        
        if seed == 'original':
            model_path = MODEL_CONFIG['PATH']
            is_original = True
        else:
            base_path = MODEL_CONFIG['PATH'].replace('../experiments/', '../experiments_seed/')
            model_path = base_path.replace('.pth', f'_{seed}.pth')
            is_original = False
            
        prior = prior_class(total_dim=MODEL_CONFIG['TOTAL_DIM'], arr_num_classes=ARR_NUM_CLASSES, device=DEVICE, scale=SCALE, fixed_means=True)
        prior.num_classes = sum(ARR_NUM_CLASSES)                                              
        prior.num_attr = NUM_ATTR
        prior.total_dim = TOTAL_DIM
        
        model = GeneralFlow(dropout_p=MODEL_CONFIG['DROPOUT']).to(DEVICE)
        
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            prior.load_state_dict(checkpoint['prior_state_dict'])
            prior.means = checkpoint['means']
        except FileNotFoundError:
            print(f"Warning: Checkpoint not found at {model_path}. Skipping seed {seed}.")
            continue
            
        temp_results = evaluate_cyclic_consistency(model, prior, DEVICE, is_original=is_original)
        
        for temp, agg_ratio in temp_results.items():
            all_results.append({
                'temperature': temp,
                'seed': seed,
                'consistency_ratio': agg_ratio
            })

    df_raw = pd.DataFrame(all_results)
    final_rows = []
    
    for temp in TEMPERATURES:
        temp_data = df_raw[df_raw['temperature'] == temp]['consistency_ratio'].values
        
        for _, row in df_raw[df_raw['temperature'] == temp].iterrows():
            final_rows.append({
                'temperature': temp,
                'seed': row['seed'],
                'consistency_ratio': row['consistency_ratio'],
                'ci_margin': np.nan 
            })
            
        mean_val, h_val = get_confidence_interval(temp_data)
        final_rows.append({
            'temperature': temp,
            'seed': 'aggregate',
            'consistency_ratio': mean_val,
            'ci_margin': h_val
        })
        
    df_final = pd.DataFrame(final_rows)
    df_final.to_csv(RESULTS_CSV, index=False)
    print(f"\nEvaluation complete. Aggregated metrics with CIs saved to '{RESULTS_CSV}'.")