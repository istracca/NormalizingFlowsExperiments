import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import itertools
import importlib
from torch.utils.data import DataLoader, TensorDataset
import random
import sys

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.join('..', '..')))
from utils import set_seed
sys.path.append(os.path.join(os.path.dirname(__file__), '../priors'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))

# --- Prior Imports ---
from SimpleSplitGMM import SimpleSplitGMM
from CheckerboardGMM import CheckerboardGMM
from SimpleSplitIB import SimpleSplitIB
from CheckerboardIB import CheckerboardIB

# ==========================================
# 1. CONFIGURATION GRID
# ==========================================
# Params not relevant to a specific prior (e.g. BETA for GMM) will be ignored.
HYPERPARAMS = {
    "SCALE": [1.0],              
    "MODEL": ["hybrid_v3_1x1_double"],      
    "PRIOR": ["CheckerboardGMM", "CheckerboardIB"],
    "BETA": [0.01, 0.1],                 # Only used for IB priors
    "OPTIMIZER": ["Adam"],
    "TRANSFORM": [0.25, 0.5],
    "DROPOUT": [0.1],
    "TYPE": ["best_loss"],
    "VERSION": ["2_attr", "4_attr"], 
    "FIXED_MEANS": [False]          # Only used for IB priors
}

# Settings
BATCH_SIZE = 128
N_RUNS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_CSV = "my_results.csv"
PLOT_BASE_DIR = "plots/samples"
TOTAL_DIM = 3 * 28 * 56  # 4704

os.makedirs(PLOT_BASE_DIR, exist_ok=True)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_confidence_interval(data_array, confidence=0.95):
    """Calculates mean and error margin for CI."""
    if len(data_array) < 2:
        return np.mean(data_array), 0.0
    mean = np.mean(data_array)
    std_err = stats.sem(data_array)
    h = std_err * stats.t.ppf((1 + confidence) / 2., len(data_array) - 1)
    return mean, h

def evaluate_model(model, prior, loader, device, n_runs):
    """
    Computes Loss and Per-Attribute Accuracy.
    Handles both GMM (scalar loss) and IB (tuple loss) return types.
    """
    model.eval()
    prior.eval()
    
    num_attr = len(prior.means)
    
    # Initialize lists
    losses = []
    gen_losses = [] # Only for IB
    cls_losses = [] # Only for IB
    
    attr_accs = {k: [] for k in range(num_attr)}
    overall_accs = []
    
    # Check if prior is IB based on class name or attribute presence
    is_ib_prior = 'IB' in prior.__class__.__name__
    
    for _ in range(n_runs):
        run_loss = 0.0
        run_gen_loss = 0.0
        run_cls_loss = 0.0
        
        correct_per_attr = np.zeros(num_attr)
        run_correct_all = 0
        run_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Dequantization
                batch_X = (batch_X * 255.0 + torch.rand_like(batch_X)) / 256.0
                batch_X = batch_X - 0.5

                # Forward Pass
                z, sldj = model(batch_X)
                
                # Loss Calculation
                loss_output = prior.get_loss(z, sldj, batch_y)
                
                if isinstance(loss_output, tuple):
                    # IB Style: (total, gen, cls)
                    loss, gen_loss, cls_loss = loss_output
                    run_loss += loss.item()
                    run_gen_loss += gen_loss.item()
                    run_cls_loss += cls_loss.item()
                else:
                    # GMM Style: scalar
                    loss = loss_output
                    run_loss += loss.item()
                    # Keep gen/cls 0.0 for GMM
                
                # Classification
                z_flat = z.view(z.size(0), -1)
                preds, _ = prior.classify(z_flat)
                
                if isinstance(preds, list):
                    preds = torch.stack(preds, dim=1)
                elif isinstance(preds, tuple): 
                    preds = torch.stack(preds[0], dim=1) if isinstance(preds[0], list) else preds[0]
                
                # Accuracy
                for k in range(num_attr):
                    matched_k = (preds[:, k] == batch_y[:, k])
                    correct_per_attr[k] += matched_k.sum().item()
                
                matched_all = (preds == batch_y).all(dim=1)
                run_correct_all += matched_all.sum().item()
                run_total += batch_y.size(0)
        
        losses.append(run_loss / len(loader))
        overall_accs.append(run_correct_all / run_total)
        for k in range(num_attr):
            attr_accs[k].append(correct_per_attr[k] / run_total)

        if is_ib_prior:
            gen_losses.append(run_gen_loss / len(loader))
            cls_losses.append(run_cls_loss / len(loader))
        else:
            gen_losses.append(np.nan)
            cls_losses.append(np.nan)
            
    return overall_accs, attr_accs, losses, gen_losses, cls_losses

def postprocess(img_tensor):
    img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(img, 0, 1)

def generate_plots(model, prior, device, config_str, save_dir, X_test_tensor, y_test_tensor):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    num_attr = len(prior.means)
    
    def build_latent(parts_list):
        current_len = len(parts_list)
        if current_len < num_attr:
            for k in range(current_len, num_attr):
                # Handle cases where means might be ParameterList or Tensor
                if isinstance(prior.means, nn.ParameterList):
                    default_mean = prior.means[k][0].view(-1)
                else:
                    default_mean = prior.means[k][0].view(-1)
                parts_list.append(default_mean.unsqueeze(0))
        return prior.get_full_latent(parts_list)

    # 1. Means Grid
    print(f"   Generating Means Grid...")
    # Safe access to means shape depending on implementation
    if isinstance(prior.means, nn.ParameterList):
        num_rows = prior.means[0].shape[0]
        num_cols = prior.means[1].shape[0]
    else:
        num_rows = prior.means[0].shape[0]
        num_cols = prior.means[1].shape[0]

    fig1, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    with torch.no_grad():
        for i in range(num_rows):
            for j in range(num_cols):
                # Construct latent from means
                mean_0 = prior.means[0][i].view(-1).unsqueeze(0)
                mean_1 = prior.means[1][j].view(-1).unsqueeze(0)
                z = build_latent([mean_0, mean_1])
                
                z_struct = z.view(1, 12, 14, 28)
                img_gen = model.inverse(z_struct)
                img_gen = img_gen + 0.5
                
                ax = axes[i, j] if num_rows > 1 and num_cols > 1 else axes
                if isinstance(ax, np.ndarray): 
                    if ax.ndim == 2: ax = ax[i,j]
                    else: ax = ax[max(i,j)] # fallback
                
                ax.imshow(postprocess(img_gen))
                ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{config_str}_zero_temp.png")
    plt.close(fig1)

    # 2. Hybrid Reconstructions (Shared Logic)
    set_seed(42)
    print(f"   Generating Hybrids...")
    num_samples = 10
    fig2, axes = plt.subplots(num_samples, 5, figsize=(10, 2 * num_samples))
    
    for row in range(num_samples):
        idx1, idx2 = np.random.choice(len(X_test_tensor), 2, replace=False)
        img1 = X_test_tensor[idx1].unsqueeze(0).to(device)
        img2 = X_test_tensor[idx2].unsqueeze(0).to(device)
        lbl1 = y_test_tensor[idx1]
        lbl2 = y_test_tensor[idx2]

        with torch.no_grad():
            img1_in = (img1 * 255.0 + torch.rand_like(img1)) / 256.0 - 0.5
            img2_in = (img2 * 255.0 + torch.rand_like(img2)) / 256.0 - 0.5
            
            z1, _ = model(img1_in)
            z2, _ = model(img2_in)
            z1_chunks = prior.get_parts(z1.squeeze(0).unsqueeze(0))
            z2_chunks = prior.get_parts(z2.squeeze(0).unsqueeze(0))
            
            # Hybrid 1: Real Latent Mix (L1 + R2)
            mix_chunks = []
            for k in range(num_attr):
                # Heuristic: 0=Left, 1=Right usually
                if k % 2 == 0: mix_chunks.append(z1_chunks[k])
                else:          mix_chunks.append(z2_chunks[k])
            z_mix = prior.get_full_latent(mix_chunks)
            rec_mix = model.inverse(z_mix.view(1, 12, 14, 28)) + 0.5
            
            # Hybrid 2: Real + Prior Mean (L1 + R2_mean)
            mix_gen_chunks = []
            for k in range(num_attr):
                if k % 2 == 0: 
                    mix_gen_chunks.append(z1_chunks[k])
                else:
                    target = lbl2[k]
                    mix_gen_chunks.append(prior.means[k][target].view(-1).unsqueeze(0))
            z_mix_gen = prior.get_full_latent(mix_gen_chunks)
            rec_mix_gen = model.inverse(z_mix_gen.view(1, 12, 14, 28)) + 0.5

            # Hybrid 3: Prior Mean + Real (L1_mean + R2)
            mix_gen_chunks_2 = []
            for k in range(num_attr):
                if k % 2 == 0: 
                    target = lbl1[k]
                    mix_gen_chunks_2.append(prior.means[k][target].view(-1).unsqueeze(0))
                else:
                    mix_gen_chunks_2.append(z2_chunks[k])
            z_mix_gen_2 = prior.get_full_latent(mix_gen_chunks_2)
            rec_mix_gen_2 = model.inverse(z_mix_gen_2.view(1, 12, 14, 28)) + 0.5

        axes[row, 0].imshow(postprocess(img1)); axes[row, 0].set_title(f"I1: {lbl1.cpu().numpy()}")
        axes[row, 1].imshow(postprocess(img2)); axes[row, 1].set_title(f"I2: {lbl2.cpu().numpy()}")
        axes[row, 2].imshow(postprocess(rec_mix)); axes[row, 2].set_title("Mix Latents")
        axes[row, 3].imshow(postprocess(rec_mix_gen)); axes[row, 3].set_title("L1 + Mean2")
        axes[row, 4].imshow(postprocess(rec_mix_gen_2)); axes[row, 4].set_title("Mean1 + R2")
        for ax in axes[row]: ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{config_str}_hybrid.png")
    plt.close(fig2)

# ==========================================
# 3. MAIN LOOP
# ==========================================

keys, values = zip(*HYPERPARAMS.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
results_by_group = {}
loaded_datasets = {}

set_seed(42)

for config in combinations:
    version = config['VERSION']
    prior_name = config['PRIOR']
    is_ib = 'IB' in prior_name
    
    # Construct paths based on model type to locate files
    # Assuming standard folder structure: experiments/models/GMM/... or experiments/models/IB/...
    folder_type = "IB" if is_ib else "GMM"
    
    # Construct config ID string
    # Note: Ensure this matches exactly how your training script names files
    if is_ib:
        config_id = f"{config['TYPE']}_{config['SCALE']}_{config['MODEL']}_{prior_name}_{config['BETA']}_{config['OPTIMIZER']}_{config['TRANSFORM']}_{config['DROPOUT']}_{config['FIXED_MEANS']}"
    else:
        config_id = f"{config['TYPE']}_{config['SCALE']}_{config['MODEL']}_{prior_name}_{config['OPTIMIZER']}_{config['TRANSFORM']}_{config['DROPOUT']}"
    
    model_path = f"../experiments/models/{folder_type}/{version}/{config_id}.pth"
    plot_dir = os.path.join(PLOT_BASE_DIR, prior_name, version)
    
    print(f"\nProcessing: {config_id} [{version}]")
    
    if not os.path.exists(model_path):
        print(f"  [!] Checkpoint not found: {model_path}")
        continue

    # --- Load Data (Cached) ---
    if version not in loaded_datasets:
        print(f"  Loading Data for {version}...")
        if version == '2_attr':
            data_path = '../data/easy_colored_double_mnist.npz'
            arr_num_classes = [10, 10]
        elif version == '4_attr':
            data_path = '../data/easy_colored_double_mnist_with_attributes.npz'
            arr_num_classes = [10, 10, 7, 7]
        
        data = np.load(data_path)
        X_val_t = torch.tensor(data['X_val'].transpose(0, 3, 1, 2), dtype=torch.float32)
        y_val_t = torch.tensor(data['y_val'], dtype=torch.long)
        X_test_t = torch.tensor(data['X_test'].transpose(0, 3, 1, 2), dtype=torch.float32)
        y_test_t = torch.tensor(data['y_test'], dtype=torch.long)
        
        loaded_datasets[version] = {
            'val_loader': DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False),
            'test_loader': DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False),
            'X_test_tensor': X_test_t,
            'y_test_tensor': y_test_t,
            'arr_num_classes': arr_num_classes
        }
    
    current_data = loaded_datasets[version]
    arr_num_classes = current_data['arr_num_classes']
    num_attr = len(arr_num_classes)

    # --- Load Model ---
    try:
        # Load Flow
        module = importlib.import_module(config['MODEL'])
        GeneralFlow = getattr(module, 'GeneralFlow')
        model = GeneralFlow(dropout_p=config['DROPOUT']).to(DEVICE)
        
        # Load Prior
        prior_class = globals()[prior_name]
        
        # Prepare Args dynamically
        prior_args = {
            'total_dim': TOTAL_DIM, 
            'arr_num_classes': arr_num_classes, 
            'device': DEVICE, 
            'scale': config['SCALE'], 
            'fixed_means': config['FIXED_MEANS'] if is_ib else False # GMM implies False usually
        }
        if is_ib:
            prior_args['beta'] = config['BETA']
            
        prior = prior_class(**prior_args).to(DEVICE)
        
        # Load Weights
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        prior.load_state_dict(checkpoint['prior_state_dict'])
        
        # Handle means loading if separate
        if 'means' in checkpoint and not isinstance(prior.means, nn.ParameterList):
             prior.means = checkpoint['means']
             
        print(f"  Loaded epoch: {checkpoint.get('epoch', 'Unknown')}")
        
    except Exception as e:
        print(f"  [!] Error loading model architecture: {e}")
        continue

    # --- Evaluate ---
    print("  Evaluating Validation Set...")
    val_acc_all, val_acc_attr, val_losses, val_gen, val_cls = evaluate_model(
        model, prior, current_data['val_loader'], DEVICE, n_runs=N_RUNS
    )
    val_metrics = {
        "val_acc": get_confidence_interval(val_acc_all),
        "val_loss": get_confidence_interval(val_losses),
        "val_gen_loss": get_confidence_interval(val_gen),
        "val_cls_loss": get_confidence_interval(val_cls),
    }

    print("  Evaluating Test Set...")
    test_acc_all, test_acc_attr, test_losses, test_gen, test_cls = evaluate_model(
        model, prior, current_data['test_loader'], DEVICE, n_runs=N_RUNS
    )
    test_metrics = {
        "test_acc": get_confidence_interval(test_acc_all),
        "test_loss": get_confidence_interval(test_losses),
        "test_gen_loss": get_confidence_interval(test_gen),
        "test_cls_loss": get_confidence_interval(test_cls),
    }
    
    print(f"  -> Val Acc: {val_metrics['val_acc'][0]:.4f}")
    
    # --- Plots ---
    print("  Generating Plots...")
    try:
        generate_plots(model, prior, DEVICE, config_id, plot_dir, 
                       current_data['X_test_tensor'], current_data['y_test_tensor'])
    except Exception as e:
        print(f"  [!] Plot warning: {e}")

    # --- Aggregate Results ---
    result_entry = config.copy()
    
    # Add Standard Metrics
    for prefix, metrics in [("val", val_metrics), ("test", test_metrics)]:
        for key, (mean, ci) in metrics.items():
            result_entry[f"{key}_mean"] = mean
            result_entry[f"{key}_ci"] = ci
            
    # Add Per-Attribute Metrics
    attr_names = ["L_Digit", "R_Digit", "L_Color", "R_Color"]
    for k in range(num_attr):
        col_name = attr_names[k] if k < 4 else f"Attr_{k}"
        
        v_mean, _ = get_confidence_interval(val_acc_attr[k])
        result_entry[f"Val_Acc_{col_name}"] = v_mean
        
        t_mean, _ = get_confidence_interval(test_acc_attr[k])
        result_entry[f"Test_Acc_{col_name}"] = t_mean

    group_key =(prior_name, version)
    if group_key not in results_by_group:
        results_by_group[group_key] = []

    results_by_group[group_key].append(result_entry)

# ==========================================
# 4. SAVE
# ==========================================
if results_by_group:
    print("\n--- Saving Results ---")
    
    for (p_name, ver), rows in results_by_group.items():
        df = pd.DataFrame(rows)
        
        csv_filename = f"results_{p_name}_{ver}.csv"
        path = os.path.join("csv", csv_filename)
        os.makedirs("csv", exist_ok=True)
        
        # --- Column Ordering Logic (Specific to this group) ---
        # 1. Hyperparams
        base_cols = list(HYPERPARAMS.keys())
        
        # 2. Metrics
        metric_cols = [
            "val_acc_mean", "val_acc_ci", "val_loss_mean", "val_loss_ci", 
            "val_gen_mean", "val_cls_mean",
            "test_acc_mean", "test_acc_ci", "test_loss_mean", "test_loss_ci", 
            "test_gen_mean", "test_cls_mean"
        ]
        
        # 3. Dynamic Attribute Columns (found in this specific DF)
        attr_cols_val = [c for c in df.columns if c.startswith("Val_Acc_")]
        attr_cols_test = [c for c in df.columns if c.startswith("Test_Acc_")]
        
        # Combine and ensure all exist in df
        ordered_cols = base_cols + metric_cols + attr_cols_val + attr_cols_test
        
        # Filter to ensure we only select columns that actually exist 
        # (in case a hyperparam was dropped or metric is missing)
        final_cols = [c for c in ordered_cols if c in df.columns]
        
        # Save
        df = df[final_cols]
        df.to_csv(path, index=False)
        print(f"Saved: {csv_filename} ({len(df)} rows)")

else:
    print("\nNo valid configurations processed.")