import os
import sys
import torch
import random
import importlib
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse

sys.path.append(os.path.abspath(os.path.join('..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../priors')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from utils import set_seed
from IB import IB

parser = argparse.ArgumentParser(description='Train a flow-based model on MNIST.')
parser.add_argument('--beta', type=float, default=0.5, help='Beta value for Information Bottleneck prior')
args = parser.parse_args()

SCALE = 0.0
BETA = args.beta
FIXED_MEANS = False
EPOCHS_WARMUP = 20
MODEL_NAME = "hybrid_v3_1x1"
OPTIMIZER = "Adam"
TRANSFORM = 0.5
DROPOUT = 0.1
SEED = 42

NUM_CLASSES = 10
TOTAL_DIM = 784                               

MODEL_CONFIG = {
    "MODEL": MODEL_NAME, 
    "PATH": f"../experiments/models/IB/best_loss_{SCALE}_{MODEL_NAME}_IB_{BETA}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}_{FIXED_MEANS}_{EPOCHS_WARMUP}.pth",
    "PRIOR": "IB",
    "DROPOUT": DROPOUT,
    "TOTAL_DIM": TOTAL_DIM
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_CSV = f"csv/IB_{BETA}/class_cons_IB.csv"
PLOT_DIR = f"plots/IB_{BETA}/"
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
    """Parses raw predictions from the prior into a standardized tensor format."""
    if isinstance(preds_raw, tuple):
        return preds_raw[0]
    return preds_raw

def plot_failure_grid(failures, temp, save_dir, max_plots=64):
    """Plots a grid of failures, showing desired vs. classified concepts for Single MNIST."""
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
            
            img_np = f['img'].squeeze() 
            
            ax.imshow(img_np, cmap='gray')
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
    
    for temp in TEMPERATURES:
        my_samples_per_concept = 1 if temp == 0.0 else SAMPLES_PER_CONCEPT
        print(f"  -> Evaluating Temperature: {temp} ({NUM_CLASSES} classes)")
        
        failures_for_temp = []
        temp_consistencies = [] 
        
        with torch.no_grad():
            for c in range(NUM_CLASSES):
                
                z = prior.means[c].to(device).unsqueeze(0).expand(my_samples_per_concept, -1)
                
                if temp > 0:
                    z = z + torch.randn_like(z) * temp
                    
                z_structural = z.view(my_samples_per_concept, 4, 14, 14)
                img_gen = model.inverse(z_structural)
                
                z_post, _ = model(img_gen)
                z_post_flat = z_post.view(my_samples_per_concept, -1)
                
                preds_post_raw = prior.classify(z_post_flat)
                preds_post = parse_predictions(preds_post_raw)
                
                correct_all = (preds_post == c)
                
                consistency_ratio = correct_all.float().mean().item()
                temp_consistencies.append(consistency_ratio)
                
                if is_original:
                    fail_mask = ~correct_all
                    if fail_mask.any():
                        fail_indices = torch.nonzero(fail_mask, as_tuple=False).squeeze(-1)
                        for idx in fail_indices[:2]: 
                            failures_for_temp.append({
                                'img': img_gen[idx].cpu().numpy(),
                                'desired': c,
                                'pred': preds_post[idx].item()
                            })
        
        aggregate_ratio = float(np.mean(temp_consistencies))
        temp_aggregates[temp] = aggregate_ratio
        
        if is_original:
            print(f"    -> Plotting failures...")
            plot_failure_grid(failures_for_temp, temp, PLOT_DIR)
            
    return temp_aggregates


if __name__ == "__main__":
    set_seed(SEED)
    
    print("Loading Generative Classifier Architecture...")
    model_module = importlib.import_module(MODEL_CONFIG['MODEL'])
    GeneralFlow = getattr(model_module, 'GeneralFlow')
    prior_class = globals()[MODEL_CONFIG['PRIOR']]
    
    all_results = []
    
    print(f"Configuration: Single MNIST | Total Classes: {NUM_CLASSES}")
    
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
            
        prior = prior_class(total_dim=MODEL_CONFIG['TOTAL_DIM'], num_classes=NUM_CLASSES, beta=BETA, device=DEVICE, scale=SCALE, fixed_means=FIXED_MEANS)
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