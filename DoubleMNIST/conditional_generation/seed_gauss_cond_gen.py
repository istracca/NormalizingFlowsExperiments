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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join('..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../priors')))
from utils import set_seed
from GaussianPrior import GaussianPrior

SEED = 42

DATA_PATH = '../data/balanced_double_mnist.npz'
BATCH_SIZE = 128

DISC_CONFIG = {
    "MODEL": "disc_v3_double", 
    "PATH": "../experiments/models/Disc/best_loss_disc_v3_double_Adam_0.5_0.1.pth",
    "DROPOUT": 0.1
}

GEN_CONFIG = {
    "MODEL": "hybrid_v3_1x1_double",
    "PATH": "../experiments/models/Gaussian/best_loss_hybrid_v3_1x1_double_GaussianPrior_Adam_0.5_0.1.pth", 
    "DROPOUT": 0.1
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARR_NUM_CLASSES = [10, 10]
NUM_ATTR = len(ARR_NUM_CLASSES)
SAMPLES_PER_CONCEPT = 1000
TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8]

rng = np.random.default_rng(SEED)
SEEDS = ["original"] + rng.integers(0, 1000000, size=4).tolist()

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
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    fig.suptitle(f"Cyclic Consistency Failures (T={temp})", fontsize=30, fontweight='bold', y=1.02)
    
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
            ax.set_title(f"Target: {f['desired']}\nPred: {f['pred']}", color='black', fontsize=20)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/consistency_failures_T_{temp}.png", bbox_inches='tight')
    plt.close(fig)

def compute_centroids(model, loader, arr_num_classes, device):
    """
    Computes Latent Centroids for both Combinatorial and Independent approaches
    using the entire calibration (training) dataset.
    """
    model.eval()
    num_attr = len(arr_num_classes)
    
    all_z = []
    all_y = []
    
    print("    -> Pre-computing Latent Centroids...")
    with torch.no_grad():
        for batch_X, batch_y in tqdm(loader, desc="       Extracting Latents", leave=False):
            batch_X = batch_X.to(device)
            batch_X = (batch_X * 255.0 + torch.rand_like(batch_X)) / 256.0 - 0.5
            
            z, _ = model(batch_X)
            all_z.append(z.view(z.size(0), -1).cpu())                                     
            all_y.append(batch_y.cpu())
            
    all_z = torch.cat(all_z, dim=0)
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

def evaluate_cyclic_consistency(disc, gen, prior, device, mode, plot_dir, is_original=False):
    disc.eval()
    gen.eval()
    prior.eval()
    
    temp_aggregates = {}
    num_cat_attr0 = ARR_NUM_CLASSES[0]
    num_cat_attr1 = ARR_NUM_CLASSES[1]
    total_combinations = num_cat_attr0 * num_cat_attr1
    
    for temp in TEMPERATURES:
        my_samples_per_concept = 1 if temp == 0.0 else SAMPLES_PER_CONCEPT
        print(f"  -> Evaluating Temperature: {temp} ({total_combinations} combinations) - Mode [{mode.upper()}]")
        
        failures_for_temp = []
        temp_consistencies = [] 
        
        with torch.no_grad():
            for i in range(num_cat_attr0):
                for j in range(num_cat_attr1):
                    
                    if mode == 'independent':
                        mean_0 = prior.independent_means[0][i].to(device).unsqueeze(0).expand(my_samples_per_concept, -1)
                        mean_1 = prior.independent_means[1][j].to(device).unsqueeze(0).expand(my_samples_per_concept, -1)
                        z = prior.get_full_latent([mean_0, mean_1])
                        
                    elif mode == 'combinatorial':
                        flat_idx = i * num_cat_attr1 + j
                        z = prior.combinatorial_means[flat_idx].to(device).unsqueeze(0).expand(my_samples_per_concept, -1)
                    
                    if temp > 0:
                        z = z + torch.randn_like(z) * temp
                        
                    z_structural = z.view(my_samples_per_concept, 4, 14, 28)
                    img_gen = gen.inverse(z_structural)
                    
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
    
    print(f"Loading Dataset: {DATA_PATH}")
    data = np.load(DATA_PATH)
    X_train, y_train = data['X_train'], data['y_train']
    
    X_train_tensor = torch.tensor(X_train.reshape(-1, 1, 28, 56), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("\nLoading Architectures...")
    disc_module = importlib.import_module(DISC_CONFIG['MODEL'])
    PseudoResNet = getattr(disc_module, 'PseudoResNet')
    
    gen_module = importlib.import_module(GEN_CONFIG['MODEL'])
    GeneralFlow = getattr(gen_module, 'GeneralFlow')
    
    all_results_indep = []
    all_results_comb = []
    
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
            
        disc = PseudoResNet(num_classes=10, dropout_p=DISC_CONFIG['DROPOUT']).to(DEVICE)
        gen = GeneralFlow(dropout_p=GEN_CONFIG['DROPOUT']).to(DEVICE)
        prior = GaussianPrior(device=DEVICE, num_attr=NUM_ATTR).to(DEVICE)
        
        try:
            disc.load_state_dict(torch.load(disc_path, map_location=DEVICE)['model_state_dict'])
            checkpoint = torch.load(gen_path, map_location=DEVICE)
            gen.load_state_dict(checkpoint['model_state_dict'])
            prior.load_state_dict(checkpoint['prior_state_dict'])
        except FileNotFoundError as e:
            print(f"Warning: Checkpoint not found. {e}. Skipping seed {seed}.")
            continue
            
        comb_centroids, _, indep_centroids = compute_centroids(gen, train_loader, ARR_NUM_CLASSES, DEVICE)
        prior.independent_means = indep_centroids
        prior.combinatorial_means = comb_centroids
        
        for mode in ['independent', 'combinatorial']:
            plot_dir = f"plots/Gaussian/{mode}/"
            os.makedirs(plot_dir, exist_ok=True)
            
            temp_results = evaluate_cyclic_consistency(disc, gen, prior, DEVICE, mode, plot_dir, is_original=is_original)
            
            for temp, agg_ratio in temp_results.items():
                row_dict = {
                    'temperature': temp,
                    'seed': seed,
                    'consistency_ratio': agg_ratio
                }
                
                if mode == 'independent':
                    all_results_indep.append(row_dict)
                else:
                    all_results_comb.append(row_dict)

    def save_aggregated_results(all_results, mode_name):
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
        results_csv = f"csv/Gaussian/class_cons_gauss_{mode_name}.csv"
        os.makedirs(os.path.dirname(results_csv), exist_ok=True)
        df_final.to_csv(results_csv, index=False)
        print(f"Evaluation complete for {mode_name}. Metrics saved to '{results_csv}'.")

    print(f"\n[{'='*40}]\nFinalizing & Saving Results\n[{'='*40}]")
    save_aggregated_results(all_results_indep, 'independent')
    save_aggregated_results(all_results_comb, 'combinatorial')