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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join('..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../priors')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from utils import set_seed
from GaussianPrior import GaussianPrior
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='4_attr', help='Version of the dataset')
args = parser.parse_args()
VERSION = args.version
SEED = 42

if VERSION == '1_attr':
    ARR_NUM_CLASSES = [10]
    DATA_PATH = '../data/easy_colored_double_mnist.npz'
elif VERSION == '2_attr':
    ARR_NUM_CLASSES = [10, 10]
    DATA_PATH = '../data/easy_colored_double_mnist.npz'
elif VERSION == '3_attr':
    ARR_NUM_CLASSES = [10, 10, 7]
    DATA_PATH = '../data/easy_colored_double_mnist_with_attributes.npz'
elif VERSION == '4_attr':
    ARR_NUM_CLASSES = [10, 10, 7, 7]
    DATA_PATH = '../data/easy_colored_double_mnist_with_attributes.npz'
else:
    raise ValueError(f"Unknown VERSION: {VERSION}")

NUM_ATTR = len(ARR_NUM_CLASSES)
TOTAL_JOINT_CLASSES = math.prod(ARR_NUM_CLASSES)

DISC_CONFIG = {
    "MODEL": "disc_v3_double", 
    "PATH": f"../experiments/models/Disc/{VERSION}/best_loss_disc_v3_double_Adam_0.5_0.1.pth",
    "DROPOUT": 0.1
}

GEN_CONFIG = {
    "MODEL": "hybrid_v3_1x1_double",
    "PATH": f"../experiments/models/Gaussian/{VERSION}/best_loss_hybrid_v3_1x1_double_GaussianPrior_Adam_0.5_0.1.pth",
    "DROPOUT": 0.1,
    "TOTAL_DIM": 4704
}

BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    all_combinations = list(itertools.product(*[range(n) for n in ARR_NUM_CLASSES]))
    total_combinations = len(all_combinations)
    strides = [math.prod(ARR_NUM_CLASSES[i+1:]) for i in range(NUM_ATTR)]
    
    for temp in TEMPERATURES:
        my_samples_per_concept = 1 if temp == 0.0 else SAMPLES_PER_CONCEPT
        print(f"  -> Evaluating Temperature: {temp} ({total_combinations} combinations) - Mode [{mode.upper()}]")
        
        failures_for_temp = []
        temp_consistencies = [] 
        
        with torch.no_grad():
            for combo_idx, combo in enumerate(all_combinations):
                
                if mode == 'independent':
                    z_parts = []
                    for k, c in enumerate(combo):
                        mean_k = prior.independent_means[k][c].to(device).unsqueeze(0).expand(my_samples_per_concept, -1)
                        z_parts.append(mean_k)
                    z = prior.get_full_latent(z_parts)
                    
                elif mode == 'combinatorial':
                    flat_idx = sum(combo[k] * strides[k] for k in range(NUM_ATTR))
                    z = prior.combinatorial_means[flat_idx].to(device).unsqueeze(0).expand(my_samples_per_concept, -1)
                
                if temp > 0:
                    z = z + torch.randn_like(z) * temp
                    
                z_structural = z.view(my_samples_per_concept, 12, 14, 28)
                img_gen = gen.inverse(z_structural)
                
                disc_outputs = disc(img_gen)
                if not isinstance(disc_outputs, (list, tuple)):
                    disc_outputs = [disc_outputs]
                
                preds_list = [torch.argmax(logits, dim=1) for logits in disc_outputs]
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
    set_seed(SEED)
    
    print(f"Loading Dataset: {DATA_PATH} (Version: {VERSION})")
    data = np.load(DATA_PATH)
    X_train, y_train = data['X_train'], data['y_train']
    
    if VERSION == "1_attr":
        y_train = y_train[:, 0:1]
    elif VERSION == "3_attr":
        y_train = y_train[:, 0:3]
        
    X_train_tensor = torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32)
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
    
    print(f"Configuration: {VERSION} | Total Attributes: {NUM_ATTR}")
    
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
            plot_dir = f"plots/Gaussian_{VERSION}/{mode}/"
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
        results_csv = f"csv/Gaussian_{VERSION}/{mode_name}/class_cons_gauss.csv"
        os.makedirs(os.path.dirname(results_csv), exist_ok=True)
        df_final.to_csv(results_csv, index=False)
        print(f"Evaluation complete for {mode_name}. Metrics saved to '{results_csv}'.")

    print(f"\n[{'='*40}]\nFinalizing & Saving Results\n[{'='*40}]")
    save_aggregated_results(all_results_indep, 'independent')
    save_aggregated_results(all_results_comb, 'combinatorial')