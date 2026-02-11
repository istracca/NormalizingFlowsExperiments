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

sys.path.append(os.path.abspath(os.path.join('..', '..')))
from utils import set_seed
sys.path.append(os.path.join(os.path.dirname(__file__), '../priors'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))
from SimpleSplitGMM import SimpleSplitGMM
from CheckerboardGMM import CheckerboardGMM

# ==========================================
# 1. CONFIGURATION GRID
# ==========================================
HYPERPARAMS = {
    "SCALE": [1.0,2.0,3.0],              
    "MODEL": ["hybrid_v3_1x1_double"],      
    "PRIOR": ["SimpleSplitGMM", "CheckerboardGMM"], 
    "OPTIMIZER": ["Adam"],
    "TRANSFORM": [0.5],
    "DROPOUT": [0.1,0.2],
    "TYPE": ["best_loss", "best_acc"],
    "VERSION": ["2_attr", "4_attr"]  # 2_attr (Digits) or 4_attr (Digits+Colors)
}

# Settings
BATCH_SIZE = 128
N_RUNS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_CSV = "results.csv"
PLOT_BASE_DIR = "plots/samples"
TOTAL_DIM = 3 * 28 * 56  # 4704 (RGB Colored Double MNIST)

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
    Returns:
      - accuracies_dict: { 'all': [run1, run2...], 'attr0': [...], 'attr1': [...] }
      - losses: [run1, run2...]
    """
    model.eval()
    prior.eval()
    
    num_attr = len(prior.means)
    
    # Initialize lists to store results for each run
    losses = []
    
    # attr_accs[k] will be a list of accuracies for attribute k across runs
    attr_accs = {k: [] for k in range(num_attr)}
    overall_accs = []
    
    for _ in range(n_runs):
        run_loss = 0.0
        
        correct_per_attr = np.zeros(num_attr)
        run_correct_all = 0
        run_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Dequantization, Normalization, Centering
                batch_X = (batch_X * 255.0 + torch.rand_like(batch_X)) / 256.0
                batch_X = batch_X - 0.5

                # Forward Pass
                z, sldj = model(batch_X)
                
                # Loss
                loss = prior.get_loss(z, sldj, batch_y)
                run_loss += loss.item()
                
                # Classification
                z_flat = z.view(z.size(0), -1)
                preds, _ = prior.classify(z_flat)
                
                if isinstance(preds, list):
                    preds = torch.stack(preds, dim=1)
                elif isinstance(preds, tuple): 
                    preds = torch.stack(preds[0], dim=1) if isinstance(preds[0], list) else preds[0]
                
                # Check accuracy for each attribute individually
                for k in range(num_attr):
                    matched_k = (preds[:, k] == batch_y[:, k])
                    correct_per_attr[k] += matched_k.sum().item()
                
                # Check Overall Accuracy
                matched_all = (preds == batch_y).all(dim=1)
                run_correct_all += matched_all.sum().item()
                run_total += batch_y.size(0)
        
        losses.append(run_loss / len(loader))
        overall_accs.append(run_correct_all / run_total)
        
        for k in range(num_attr):
            attr_accs[k].append(correct_per_attr[k] / run_total)
            
    return overall_accs, attr_accs, losses

def postprocess(img_tensor):
    img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(img, 0, 1)

def generate_plots(model, prior, device, config_str, save_dir, X_test_tensor, y_test_tensor):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    num_attr = len(prior.means)
    total_dim = 4704
    dims_per_attr = total_dim // num_attr
    
    def build_latent(parts_list):
        current_len = len(parts_list)
        if current_len < num_attr:
            for k in range(current_len, num_attr):
                default_mean = prior.means[k][0].view(-1)
                parts_list.append(default_mean.unsqueeze(0))
        return prior.get_full_latent(parts_list)

    # 1. Means Grid
    print(f"   Generating Means Grid...")
    num_rows = prior.means[0].shape[0]
    num_cols = prior.means[1].shape[0]
    
    fig1, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    with torch.no_grad():
        for i in range(num_rows):
            for j in range(num_cols):
                z = build_latent([prior.means[0][i].view(-1).unsqueeze(0), 
                                  prior.means[1][j].view(-1).unsqueeze(0)])
                z_struct = z.view(1, 12, 14, 28)
                img_gen = model.inverse(z_struct)
                img_gen = img_gen + 0.5
                ax = axes[i, j]
                ax.imshow(postprocess(img_gen))
                ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{config_str}_zero_temp.png")
    plt.close(fig1)

    # 2. Hybrid Reconstructions
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
            img1_in = (img1 * 255.0 + torch.rand_like(img1)) / 256.0
            img1_in = img1_in - 0.5
            img2_in = (img2 * 255.0 + torch.rand_like(img2)) / 256.0
            img2_in = img2_in - 0.5
            
            z1, _ = model(img1_in)
            z2, _ = model(img2_in)
            z1_flat = z1.squeeze(0)
            z2_flat = z2.squeeze(0)

            z1_chunks = prior.get_parts(z1_flat.unsqueeze(0))
            z2_chunks = prior.get_parts(z2_flat.unsqueeze(0))
            
            # Hybrid 1: Real Latent Mix
            mix_chunks = []
            for k in range(num_attr):
                if k % 2 == 0: mix_chunks.append(z1_chunks[k])
                else:          mix_chunks.append(z2_chunks[k])
            z_mix = prior.get_full_latent(mix_chunks)
            rec_mix = model.inverse(z_mix.view(1, 12, 14, 28))
            rec_mix = rec_mix + 0.5
            
            # Hybrid 2: Real + Prior Mean
            mix_gen_chunks = []
            for k in range(num_attr):
                if k % 2 == 0: 
                    mix_gen_chunks.append(z1_chunks[k])
                else:
                    target = lbl2[k]
                    mix_gen_chunks.append(prior.means[k][target].view(-1).unsqueeze(0))
            z_mix_gen = prior.get_full_latent(mix_gen_chunks)
            rec_mix_gen = model.inverse(z_mix_gen.view(1, 12, 14, 28))
            rec_mix_gen = rec_mix_gen + 0.5

            # Hybrid 3: Prior Mean + Real
            mix_gen_chunks_2 = []
            for k in range(num_attr):
                if k % 2 == 0: 
                    target = lbl1[k]
                    mix_gen_chunks_2.append(prior.means[k][target].view(-1).unsqueeze(0))
                else:
                    mix_gen_chunks_2.append(z2_chunks[k])
            z_mix_gen_2 = prior.get_full_latent(mix_gen_chunks_2)
            rec_mix_gen_2 = model.inverse(z_mix_gen_2.view(1, 12, 14, 28))
            rec_mix_gen_2 = rec_mix_gen_2 + 0.5


        axes[row, 0].imshow(postprocess(img1))
        axes[row, 0].set_title(f"Image 1: {lbl1.cpu().numpy()}")
        axes[row, 1].imshow(postprocess(img2))
        axes[row, 1].set_title(f"Image 2: {lbl2.cpu().numpy()}")
        axes[row, 2].imshow(postprocess(rec_mix))
        axes[row, 2].set_title("Hybrid: L1+R2")
        axes[row, 3].imshow(postprocess(rec_mix_gen))
        axes[row, 3].set_title("Hybrid: L1+R2n")
        axes[row, 4].imshow(postprocess(rec_mix_gen_2))
        axes[row, 4].set_title("Hybrid: L1n+R2")
        for ax in axes[row]: ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{config_str}_hybrid.png")
    plt.close(fig2)


# ==========================================
# 3. MAIN LOOP
# ==========================================

keys, values = zip(*HYPERPARAMS.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
results = []
loaded_datasets = {}

set_seed(42)

for config in combinations:
    version = config['VERSION']
    config_id = f"{config['TYPE']}_{config['SCALE']}_{config['MODEL']}_{config['PRIOR']}_{config['OPTIMIZER']}_{config['TRANSFORM']}_{config['DROPOUT']}"
    
    model_path = f"../experiments/models/GMM/{version}/{config_id}.pth"
    plot_dir = os.path.join(PLOT_BASE_DIR, version)
    
    print(f"\nProcessing: {config_id} [{version}]")
    
    if not os.path.exists(model_path):
        print(f"  [!] Checkpoint not found: {model_path}. Skipping.")
        continue

    # --- Load Data ---
    if version not in loaded_datasets:
        print(f"  Loading Data for {version}...")
        if version == '2_attr':
            data_path = '../data/colored_double_mnist.npz'
            arr_num_classes = [10, 10]
        elif version == '4_attr':
            data_path = '../data/colored_double_mnist_with_attributes.npz'
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
        module = importlib.import_module(config['MODEL'])
        GeneralFlow = getattr(module, 'GeneralFlow')
        
        prior_class = globals()[config['PRIOR']]        
        prior = prior_class(
            total_dim=TOTAL_DIM, 
            arr_num_classes=arr_num_classes, 
            num_attr=num_attr, 
            device=DEVICE, 
            scale=config['SCALE'], 
            fixed_means=False
        ).to(DEVICE)
        
        model = GeneralFlow(dropout_p=config['DROPOUT']).to(DEVICE)
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        prior.load_state_dict(checkpoint['prior_state_dict'])
        
        if 'means' in checkpoint and not isinstance(prior.means, nn.ParameterList):
             prior.means = checkpoint['means']
             
        print(f"  Loaded epoch: {checkpoint.get('epoch', 'Unknown')}")
        
    except Exception as e:
        print(f"  [!] Error loading model: {e}")
        continue

    # --- Evaluate ---
    print("  Evaluating Validation Set...")

    val_acc_all, val_acc_attr, val_losses = evaluate_model(model, prior, current_data['val_loader'], DEVICE, n_runs=N_RUNS)
    
    val_acc_mean, val_acc_ci = get_confidence_interval(val_acc_all)
    val_loss_mean, val_loss_ci = get_confidence_interval(val_losses)
    
    print("  Evaluating Test Set...")
    test_acc_all, test_acc_attr, test_losses = evaluate_model(model, prior, current_data['test_loader'], DEVICE, n_runs=N_RUNS)
    
    test_acc_mean, test_acc_ci = get_confidence_interval(test_acc_all)
    test_loss_mean, test_loss_ci = get_confidence_interval(test_losses)
    
    print(f"  -> Val Acc: {val_acc_mean:.4f} ± {val_acc_ci:.4f}")
    print(f"  -> Val Loss: {val_loss_mean:.4f} ± {val_loss_ci:.4f}")
    
    # --- Plot ---
    print("  Generating Plots...")
    try:
        generate_plots(
            model, prior, DEVICE, config_id, plot_dir, 
            current_data['X_test_tensor'], current_data['y_test_tensor']
        )
    except Exception as e:
        print(f"  [!] Plot error: {e}")

    # --- Save Results ---
    result_entry = config.copy()
    
    # 1. Basic Metrics
    result_entry.update({
        "val_acc_mean": val_acc_mean,
        "val_acc_ci": val_acc_ci,
        "val_loss_mean": val_loss_mean,
        "val_loss_ci": val_loss_ci,
        "test_acc_mean": test_acc_mean,
        "test_acc_ci": test_acc_ci,
        "test_loss_mean": test_loss_mean,
        "test_loss_ci": test_loss_ci
    })
    
    # 2. Per-Attribute Metrics
    # Attr 0 = Left Digit, Attr 1 = Right Digit
    # Attr 2 = Left Color, Attr 3 = Right Color (if 4_attr)
    attr_names = ["L_Digit", "R_Digit", "L_Color", "R_Color"]
    
    for k in range(num_attr):
        mean_acc, _ = get_confidence_interval(val_acc_attr[k])
        col_name = attr_names[k] if k < 4 else f"Attr_{k}"
        result_entry[f"Val_Acc_{col_name}"] = mean_acc

    for k in range(num_attr):
        mean_acc, _ = get_confidence_interval(test_acc_attr[k])
        # Use safe name if k < 4, else generic
        col_name = attr_names[k] if k < 4 else f"Attr_{k}"
        result_entry[f"Test_Acc_{col_name}"] = mean_acc

    results.append(result_entry)

# ==========================================
# 4. WRITE TO CSV
# ==========================================
if results:
    df = pd.DataFrame(results)
    
    # Reorder columns: Hyperparams first, then Exact Acc, then Per-Attribute Acc
    base_cols = list(HYPERPARAMS.keys()) + ["val_acc_mean", "val_acc_ci", "val_loss_mean", "val_loss_ci",
                                            "test_acc_mean", "test_acc_ci", "test_loss_mean", "test_loss_ci"]

    attr_cols_val = [c for c in df.columns if c.startswith("Val_Acc_") and c not in base_cols]
    attr_cols_test = [c for c in df.columns if c.startswith("Test_Acc_") and c not in base_cols]
    
    # Combined column order
    # Use list(dict.fromkeys(...)) to remove duplicates while preserving order
    final_cols = list(dict.fromkeys(base_cols + attr_cols_val + attr_cols_test))
    
    df = df[final_cols]
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nCompleted. Results saved to {RESULTS_CSV}")
else:
    print("\nNo valid configurations processed.")