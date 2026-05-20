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
import argparse

sys.path.append(os.path.abspath(os.path.join('..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from utils import set_seed

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='4_attr', help='Version of the dataset')
args = parser.parse_args()
VERSION = args.version
SEED = 42

if VERSION == '1_attr':
    ARR_NUM_CLASSES = [10]
elif VERSION == '2_attr':
    ARR_NUM_CLASSES = [10, 10]
elif VERSION == '3_attr':
    ARR_NUM_CLASSES = [10, 10, 7]
elif VERSION == '4_attr':
    ARR_NUM_CLASSES = [10, 10, 7, 7]
else:
    raise ValueError(f"Unknown VERSION: {VERSION}")

NUM_ATTR = len(ARR_NUM_CLASSES)
COND_DIM = 64

DISC_CONFIG = {
    "MODEL": "disc_v3_double", 
    "PATH": f"../experiments/models/Disc/{VERSION}/best_loss_disc_v3_double_Adam_0.5_0.1.pth",
    "DROPOUT": 0.1
}

GEN_CONFIG = {
    "MODEL": "conditional_scale",
    "PATH": f"../experiments/models/Conditional/{VERSION}/best_loss_conditional_scale_GaussianPrior_Adam_0.5_0.1.pth",
    "DROPOUT": 0.1
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_CSV = f"csv/Conditional_{VERSION}/class_cons_cond.csv"
PLOT_DIR = f"plots/Conditional_{VERSION}/"
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
    """Plots a grid of failures, showing desired vs. classified concepts for Colored MNIST."""
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
            
            img_np = f['img'] + 0.5 
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = np.clip(img_np, 0, 1)
            
            ax.imshow(img_np)
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
    
    all_combinations = list(itertools.product(*[range(n) for n in ARR_NUM_CLASSES]))
    total_combinations = len(all_combinations)
    
    for temp in TEMPERATURES:
        my_samples_per_concept = 1 if temp == 0.0 else SAMPLES_PER_CONCEPT
        print(f"  -> Evaluating Temperature: {temp} ({total_combinations} combinations)")
        
        failures_for_temp = []
        temp_consistencies = [] 
        
        with torch.no_grad():
            for combo_idx, combo in enumerate(all_combinations):
                
                y_onehot_list = []
                for k, c in enumerate(combo):
                    lbls = torch.full((my_samples_per_concept,), c, dtype=torch.long, device=device)
                    num_classes_for_attr = ARR_NUM_CLASSES[k]
                    y_onehot = F.one_hot(lbls, num_classes=num_classes_for_attr).float()
                    y_onehot_list.append(y_onehot)
                
                y_onehot_cond = torch.cat(y_onehot_list, dim=1)
                
                if temp > 0:
                    z = torch.randn(my_samples_per_concept, 12, 14, 28, device=device) * temp
                else:
                    z = torch.zeros(my_samples_per_concept, 12, 14, 28, device=device)
                    
                img_gen = gen.inverse(z, y_onehot_cond)
                
                logits_tuple = disc(img_gen)
                
                if not isinstance(logits_tuple, (list, tuple)):
                    logits_tuple = (logits_tuple,)
                
                preds_list = [torch.argmax(logits, dim=1) for logits in logits_tuple]
                preds_post = torch.stack(preds_list, dim=1)
                
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
    set_seed(42)
    
    print("Loading Architectures...")
    disc_module = importlib.import_module(DISC_CONFIG['MODEL'])
    PseudoResNet = getattr(disc_module, 'PseudoResNet')
    
    gen_module = importlib.import_module(GEN_CONFIG['MODEL'])
    GeneralFlow = getattr(gen_module, 'GeneralFlow')
    
    all_results = []
    
    print(f"Starting Cyclic Consistency Evaluation...")
    print(f"Configuration: {VERSION} | Total Attributes: {NUM_ATTR} | Conditional Dim: {COND_DIM}")
    
    for seed in SEEDS:
        print(f"\n[{'='*40}]")
        print(f"Processing Model Seed: {seed}")
        print(f"[{'='*40}]")
        
        if seed == 'original':
            disc_path = DISC_CONFIG['PATH']
            gen_path = GEN_CONFIG['PATH']
            is_original = True
        else:
            disc_base = DISC_CONFIG['PATH'].replace('../experiments/', '../experiments_seed/')
            disc_path = disc_base.replace('.pth', f'_{seed}.pth')
            
            gen_base = GEN_CONFIG['PATH'].replace('../experiments/', '../experiments_seed/')
            gen_path = gen_base.replace('.pth', f'_{seed}.pth')
            is_original = False
            
        disc = PseudoResNet(arr_num_classes=ARR_NUM_CLASSES, in_channels=3, dropout_p=DISC_CONFIG['DROPOUT'], device=DEVICE).to(DEVICE)
        
        gen = GeneralFlow(num_classes=sum(ARR_NUM_CLASSES), dropout_p=GEN_CONFIG['DROPOUT'], cond_dim=COND_DIM).to(DEVICE)
        
        try:
            disc.load_state_dict(torch.load(disc_path, map_location=DEVICE)['model_state_dict'])
            gen.load_state_dict(torch.load(gen_path, map_location=DEVICE)['model_state_dict'])
        except FileNotFoundError as e:
            print(f"Warning: Checkpoint not found. {e}. Skipping seed {seed}.")
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