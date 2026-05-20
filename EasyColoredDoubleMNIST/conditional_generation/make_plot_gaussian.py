import os
import sys
import torch
import importlib
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import argparse

sys.path.append(os.path.abspath(os.path.join('..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../priors')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from utils import set_seed
from GaussianPrior import GaussianPrior

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='4_attr', choices=['2_attr', '4_attr'], 
                    help='Dataset version to evaluate')
args = parser.parse_args()

VERSION = args.version
APPROACHES = ['indep.', 'comb.']
TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8]

SAMPLES_PER_TEMP = {
    0.0: 1,
    0.2: 3,
    0.4: 3,
    0.6: 3,
    0.8: 3
}

if VERSION == '2_attr':
    ARR_NUM_CLASSES = [10, 10]
    TARGETS = [(0, 1), (2, 3), (4, 5)]
    DATA_PATH = '../data/easy_colored_double_mnist.npz'
elif VERSION == '4_attr':
    ARR_NUM_CLASSES = [10, 10, 7, 7]
    TARGETS = [(0, 1, 0, 1), (2, 3, 2, 3), (4, 5, 4, 5)]
    DATA_PATH = '../data/easy_colored_double_mnist_with_attributes.npz'

NUM_ATTR = len(ARR_NUM_CLASSES)
TOTAL_DIM = 4704 
BATCH_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = f"samples_plots/Gaussian/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_centroids(model, loader, arr_num_classes, device):
    """
    Computes Latent Centroids for both Combinatorial and Independent approaches
    using the entire calibration (training) dataset.
    """
    model.eval()
    num_attr = len(arr_num_classes)
    
    all_z = []
    all_y = []
    
    print("\nPre-computing Latent Centroids...")
    with torch.no_grad():
        for batch_X, batch_y in tqdm(loader, desc="Extracting Latents"):
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
    
    for comb in all_combinations:
        comb_tensor = torch.tensor(comb)
        mask = (all_y == comb_tensor).all(dim=1)
        if mask.sum() > 0:
            comb_centroids.append(all_z[mask].mean(dim=0))
        else:
            comb_centroids.append(torch.full((latent_dim,), float('inf')))
            
    comb_centroids = torch.stack(comb_centroids).to(device)
    
    return comb_centroids, indep_centroids

def generate_and_plot_grid():
    set_seed(42)

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
    
    # --- Model Loading ---
    model_path = f"../experiments/models/Gaussian/{VERSION}/best_loss_hybrid_v3_1x1_double_GaussianPrior_Adam_0.5_0.1.pth"
    model_module = importlib.import_module("hybrid_v3_1x1_double")
    GeneralFlow = getattr(model_module, 'GeneralFlow')
    
    gen = GeneralFlow(dropout_p=0.1).to(DEVICE)
    prior = GaussianPrior(device=DEVICE, num_attr=NUM_ATTR).to(DEVICE)
    
    print("Loading Gaussian Generator & Prior...")
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        gen.load_state_dict(checkpoint['model_state_dict'])
        prior.load_state_dict(checkpoint['prior_state_dict'])
    except FileNotFoundError:
        print(f"Checkpoint not found at {model_path}. Exiting.")
        return
        
    comb_centroids, indep_centroids = compute_centroids(gen, train_loader, ARR_NUM_CLASSES, DEVICE)
    prior.independent_means = indep_centroids
    prior.combinatorial_means = comb_centroids
    
    gen.eval()
    prior.eval()
    
    num_approaches = len(APPROACHES)
    num_targets = len(TARGETS)
    
    width_ratios = [0.25] + [SAMPLES_PER_TEMP[t] for t in TEMPERATURES]
    total_cols_visual = sum(width_ratios)
    
    fig = plt.figure(figsize=(total_cols_visual * 2.0, num_approaches * num_targets * 1 + 1))
    fig.suptitle(f"Conditional Generation Grid - HCE", fontsize=28, fontweight='bold', y=1.00)
    
    gs_outer = gridspec.GridSpec(
        num_approaches, len(TEMPERATURES) + 1, 
        figure=fig, 
        width_ratios=width_ratios,
        wspace=0.15, 
        hspace=0.35, 
        top=0.83,
    )
    
    strides = [math.prod(ARR_NUM_CLASSES[i+1:]) for i in range(NUM_ATTR)]
    
    for a_idx, approach in enumerate(APPROACHES):
        print(f"Processing Approach: {approach.capitalize()}...")
        
        ax_approach = fig.add_subplot(gs_outer[a_idx, 0])
        ax_approach.axis('off')
        ax_approach.text(0.3, 0.5, f"{approach.capitalize()}", fontsize=24, fontweight='bold', 
                      va='center', ha='center', rotation=90)
        
        with torch.no_grad():
            for temp_idx, temp in enumerate(TEMPERATURES):
                num_samples = SAMPLES_PER_TEMP[temp]
                
                gs_inner = gridspec.GridSpecFromSubplotSpec(
                    num_targets, num_samples, 
                    subplot_spec=gs_outer[a_idx, temp_idx + 1], 
                    wspace=0.05, 
                    hspace=0.05  
                )
                
                for t_idx, target in enumerate(TARGETS):
                    
                    if approach == 'indep.':
                        z_parts = []
                        for k, c in enumerate(target):
                            mean_k = prior.independent_means[k][c].to(DEVICE).unsqueeze(0).expand(num_samples, -1)
                            z_parts.append(mean_k)
                        z = prior.get_full_latent(z_parts)
                        
                    elif approach == 'comb.':
                        flat_idx = sum(target[k] * strides[k] for k in range(NUM_ATTR))
                        z = prior.combinatorial_means[flat_idx].to(DEVICE).unsqueeze(0).expand(num_samples, -1)
                        
                    if temp > 0.0:
                        z = z + torch.randn_like(z) * temp
                        
                    z_structural = z.view(num_samples, 12, 14, 28)
                    img_gen = gen.inverse(z_structural)
                    
                    for i in range(num_samples):
                        ax = fig.add_subplot(gs_inner[t_idx, i])
                        
                        img_np = img_gen[i].cpu().numpy() + 0.5 
                        img_np = np.transpose(img_np, (1, 2, 0)) 
                        img_np = np.clip(img_np, 0, 1)
                        
                        ax.imshow(img_np)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        
                        if a_idx == 0 and t_idx == 0 and i == (num_samples // 2):
                            ax.set_title(f"T = {temp}", fontsize=22, fontweight='bold', pad=20)

    save_path = f"{OUTPUT_DIR}/cond_gen_gaussian_{VERSION}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Grid successfully generated and saved to: {save_path}")

if __name__ == "__main__":
    generate_and_plot_grid()