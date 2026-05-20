import os
import sys
import torch
import torch.nn.functional as F
import random
import importlib
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

sys.path.append(os.path.abspath(os.path.join('..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from utils import set_seed

COND_DIM = 64
SEED = 42

DISC_CONFIG = {
    "MODEL": "disc_v3_double", 
    "PATH": "../experiments/models/Disc/best_loss_disc_v3_double_Adam_0.5_0.1.pth",
    "DROPOUT": 0.1
}

GEN_CONFIG = {
    "MODEL": "conditional_scale",
    "PATH": "../experiments/models/Conditional/best_loss_conditional_scale_GaussianPrior_Adam_0.5_0.1.pth",
    "DROPOUT": 0.1
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_CSV = "csv/Conditional/class_cons_Conditional.csv"
PLOT_DIR = "plots/Conditional/"
NUM_CLASSES = 10
ARR_NUM_CLASSES = [10, 10]
NUM_ATTR = len(ARR_NUM_CLASSES)
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

def evaluate_cyclic_consistency(disc, gen, device, is_original=False):
    disc.eval()
    gen.eval()
    
    temp_aggregates = {}
    total_combinations = ARR_NUM_CLASSES[0] * ARR_NUM_CLASSES[1]
    
    for temp in TEMPERATURES:
        my_samples_per_concept = 1 if temp == 0.0 else SAMPLES_PER_CONCEPT
        print(f"  -> Evaluating Temperature: {temp} ({total_combinations} combinations)")
        
        failures_for_temp = []
        temp_consistencies = [] 
        
        with torch.no_grad():
            for i in range(ARR_NUM_CLASSES[0]):
                for j in range(ARR_NUM_CLASSES[1]):
                    
                    lbls_0 = torch.full((my_samples_per_concept,), i, dtype=torch.long, device=device)
                    lbls_1 = torch.full((my_samples_per_concept,), j, dtype=torch.long, device=device)
                    
                    y_onehot_0 = F.one_hot(lbls_0, num_classes=NUM_CLASSES).float()
                    y_onehot_1 = F.one_hot(lbls_1, num_classes=NUM_CLASSES).float()
                    y_onehot_cond = torch.cat([y_onehot_0, y_onehot_1], dim=1)
                    
                    if temp > 0:
                        z = torch.randn(my_samples_per_concept, 4, 14, 28, device=device) * temp
                    else:
                        z = torch.zeros(my_samples_per_concept, 4, 14, 28, device=device)
                        
                    img_gen = gen.inverse(z, y_onehot_cond)
                    
                    logits0, logits1 = disc(img_gen)
                    preds0 = torch.argmax(logits0, dim=1)
                    preds1 = torch.argmax(logits1, dim=1)
                    
                    preds_post = torch.stack([preds0, preds1], dim=1)
                    
                    correct_attr0 = (preds0 == i)
                    correct_attr1 = (preds1 == j)
                    correct_both = correct_attr0 & correct_attr1
                    
                    consistency_ratio = correct_both.float().mean().item()
                    temp_consistencies.append(consistency_ratio)
                    
                    if is_original:
                        fail_mask = ~correct_both
                        if fail_mask.any():
                            fail_indices = torch.nonzero(fail_mask, as_tuple=False).squeeze(-1)
                            for idx in fail_indices[:2]: 
                                failures_for_temp.append({
                                    'img': img_gen[idx].cpu().numpy(),
                                    'desired': [i, j],
                                    'pred': preds_post[idx].cpu().tolist()
                                })
        
        aggregate_ratio = float(np.mean(temp_consistencies))
        temp_aggregates[temp] = aggregate_ratio
        
            
    return temp_aggregates


if __name__ == "__main__":
    set_seed(SEED)
    
    print("Loading Architectures...")
    disc_module = importlib.import_module(DISC_CONFIG['MODEL'])
    PseudoResNet = getattr(disc_module, 'PseudoResNet')
    
    gen_module = importlib.import_module(GEN_CONFIG['MODEL'])
    GeneralFlow = getattr(gen_module, 'GeneralFlow')
    
    all_results = []
    
    print(f"Configuration: DoubleMNIST | Total Attributes: {NUM_ATTR} | Total Classes: {ARR_NUM_CLASSES}")
    
    for seed in SEEDS:
        print(f"\n[{'='*40}]")
        print(f"Processing Model Seed: {seed}")
        print(f"[{'='*40}]")
        
        if seed == 'original':
            disc_path = DISC_CONFIG['PATH']
            gen_path = GEN_CONFIG['PATH']
            is_original = True
        else:
            disc_path = DISC_CONFIG['PATH'].replace('../experiments/', '../experiments_seed/').replace('.pth', f'_{seed}.pth')
            gen_path = GEN_CONFIG['PATH'].replace('../experiments/', '../experiments_seed/').replace('.pth', f'_{seed}.pth')
            is_original = False
            
        disc = PseudoResNet(num_classes=NUM_CLASSES, dropout_p=DISC_CONFIG['DROPOUT']).to(DEVICE)
        gen = GeneralFlow(dropout_p=GEN_CONFIG['DROPOUT'], num_classes=20, cond_dim=COND_DIM).to(DEVICE)
        
        try:
            disc.load_state_dict(torch.load(disc_path, map_location=DEVICE)['model_state_dict'])
            gen.load_state_dict(torch.load(gen_path, map_location=DEVICE)['model_state_dict'])
        except FileNotFoundError:
            print(f"Warning: Checkpoints not found. Skipping seed {seed}.")
            continue
            
        temp_results = evaluate_cyclic_consistency(disc, gen, DEVICE, is_original=is_original)
        
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