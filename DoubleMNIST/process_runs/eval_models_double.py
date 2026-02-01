import os
import torch
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
from SimpleSplitGMM import SimpleSplitGMM
# ==========================================
# 1. CONFIGURATION GRID
# ==========================================
# Define the combinations you want to test here
HYPERPARAMS = {
    "SCALE": [1.0, 2.0, 3.0],              
    "MODEL": ["hybrid_v3_1x1_double"],      
    "PRIOR": ["SimpleSplitGMM"],
    "OPTIMIZER": ["Adam"],
    "TRANSFORM": [0.0, 0.25, 0.5],
    "DROPOUT": [0.0, 0.1, 0.2],
    "TYPE": ["best_loss", "best_acc"]         
}

# Settings
BATCH_SIZE = 128
N_RUNS = 50          # Number of passes for CI calculation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_CSV = "double_experiment_results.csv"
PLOT_DIR = "plots/samples"  # <--- UPDATED PATH

# Create plot directory if it doesn't exist
os.makedirs(PLOT_DIR, exist_ok=True)

# ==========================================
# 2. DATA LOADING
# ==========================================
print("Loading Data...")
data = np.load('../data/balanced_double_mnist.npz')
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']
X_test_tensor = torch.tensor(X_test.reshape(-1, 1, 28, 56), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# Prepare Tensors
def prepare_loader(X, y, batch_size, shuffle=False):
    X_tensor = torch.tensor(X.reshape(-1, 1, 28, 56), dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

val_loader = prepare_loader(X_val, y_val, BATCH_SIZE)
test_loader = prepare_loader(X_test, y_test, BATCH_SIZE)
print("Data Loaded.")

# ==========================================
# 3. HELPER FUNCTIONS
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
    Computes Accuracy and Loss using the specified logic.
    Returns arrays of results (one per run) for CI calculation.
    """
    model.eval()
    prior.eval()
    
    accuracies = []
    losses = []
    
    for _ in range(n_runs):
        run_loss = 0.0
        run_correct = 0
        run_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Dequantization
                batch_X = (batch_X * 255.0 + torch.rand_like(batch_X)) / 256.0
                batch_X = batch_X - 0.5
                
                # Forward Pass
                z, sldj = model(batch_X)
                
                # --- Loss Calculation (User Logic) ---
                loss = prior.get_loss(z, sldj, batch_y)
                run_loss += loss.item()
                
                # --- Accuracy Check ---
                z_flat = z.view(z.size(0), -1)
                preds, _ = prior.classify(z_flat)
                if isinstance(preds, list):
                    preds = torch.stack(preds, dim=1)
                elif isinstance(preds, tuple): # Handling your previous logic
                    preds = torch.stack(preds[0], dim=1) if isinstance(preds[0], list) else preds[0]
                matched_rows = (preds == batch_y).all(dim=1)
                run_correct += matched_rows.sum().item()
                run_total += batch_y.size(0)
        
        # Aggregate for this run
        avg_loss = run_loss / len(loader) # Average batch loss
        avg_acc = run_correct / run_total
        
        losses.append(avg_loss)
        accuracies.append(avg_acc)
        
    return np.array(accuracies), np.array(losses)

def generate_plots(model, prior, device, config_str, save_dir):
    """Generates and saves the 3 required plots."""
    model.eval()
    prior.eval()
    
    # 1. Zero Temperature (Means)
    num_cat_attr0 = prior.means[0].shape[0]  # e.g., 10
    num_cat_attr1 = prior.means[1].shape[0]  # e.g., 10
    temp = 0

    fig1, axes = plt.subplots(num_cat_attr0, num_cat_attr1, figsize=(num_cat_attr1, num_cat_attr0))
    with torch.no_grad():
        for i in range(num_cat_attr0):
            for j in range(num_cat_attr1):
                mean_0 = prior.means[0][i].view(-1)
                mean_1 = prior.means[1][j].view(-1)
                z = prior.get_full_latent([mean_0.unsqueeze(0), mean_1.unsqueeze(0)])
                if temp > 0:
                    z = z + torch.randn_like(z) * temp

                z_structural = z.view(1, 4, 14, 28)
                img_gen = model.inverse(z_structural)

                ax = axes[i, j]
                ax.imshow(img_gen.squeeze().cpu(), cmap='gray')
                ax.set_title(f"{i},{j}")
                ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{config_str}_zero_temp.png")
    plt.close(fig1)

    # 2. Low Temperature (0.25) Samples
    temp = 0.25
    num_versions = 5
    num_samples = 10
    combinations = [(i, j) for i in range(num_cat_attr0) for j in range(num_cat_attr1)]
    chosen_combinations = random.sample(combinations, num_samples)

    fig2, axes = plt.subplots(num_samples, num_versions, figsize=(num_versions * 2, num_samples * 1.2))
    fig2.suptitle(f"Random samples from prior means with noise (T={temp}): each row is a (digit0, digit1) pair, each column is a different noise sample", fontsize=14)

    with torch.no_grad():
        for row, (i, j) in enumerate(chosen_combinations):
            mean_0 = prior.means[0][i]
            mean_1 = prior.means[1][j]
            for col in range(num_versions):
                z = prior.get_full_latent([mean_0.unsqueeze(0), mean_1.unsqueeze(0)])
                z = z + torch.randn_like(z) * temp
                z_structural = z.view(1, 4, 14, 28)
                img_gen = model.inverse(z_structural)
                ax = axes[row, col]
                ax.imshow(img_gen.squeeze().cpu(), cmap='gray')
                if col == 0:
                    ax.set_ylabel(f"{i},{j}", fontsize=10)
                ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{config_str}_samples.png")
    plt.close(fig2)

    # 3. Interpolation (3 -> 9)
    digit_a_0 = 1
    digit_a_1 = 8
    digit_b_0 = 2
    digit_b_1 = 3
    num_steps = 11
    temp = 0.25
    fig3, axes = plt.subplots(1, num_steps, figsize=(22, 1.8))
    with torch.no_grad():
        mean_a_0 = prior.means[0][digit_a_0].unsqueeze(0).to(device)
        mean_a_1 = prior.means[1][digit_a_1].unsqueeze(0).to(device)
        mean_b_0 = prior.means[0][digit_b_0].unsqueeze(0).to(device)
        mean_b_1 = prior.means[1][digit_b_1].unsqueeze(0).to(device)
        for i, alpha in enumerate(np.linspace(0, 1, num_steps)):
            z = prior.get_full_latent([(1-alpha) * mean_a_0 + alpha * mean_b_0, (1-alpha) * mean_a_1 + alpha * mean_b_1])
            z = z + torch.randn_like(z) * temp
            z_structural = z.view(1, 4, 14, 28)
            img_gen = model.inverse(z_structural)
            ax = axes[i]
            ax.imshow(img_gen.squeeze().cpu(), cmap='gray')
            ax.set_title(f"α={alpha:.2f}")
            ax.axis('off')
    plt.suptitle(f"Interpolation between digits {digit_a_0}{digit_a_1} and {digit_b_0}{digit_b_1}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{config_str}_interp.png")
    plt.close(fig3)

    #4. Hybrid Reconstructions
    num_samples = 15
    fig4, axes = plt.subplots(num_samples, 5, figsize=(15, 2.2 * num_samples))
    set_seed(42)  # For reproducibility

    for row in range(num_samples):
        idx1, idx2 = random.sample(range(len(X_test_tensor)), 2)
        img1 = X_test_tensor[idx1].unsqueeze(0).to(device)
        img2 = X_test_tensor[idx2].unsqueeze(0).to(device)
        label1 = y_test_tensor[idx1]
        label2 = y_test_tensor[idx2]

        with torch.no_grad():
            img1_proc = (img1 * 255.0 + torch.rand_like(img1)) / 256.0 - 0.5
            img2_proc = (img2 * 255.0 + torch.rand_like(img2)) / 256.0 - 0.5

            z1, _ = model(img1_proc)
            z2, _ = model(img2_proc)
            z1_flat = z1.squeeze(0)
            z2_flat = z2.squeeze(0)

            mu_left_z1 = prior.means[0][label1[0]].to(device)
            mu_right_z2 = prior.means[1][label2[1]].to(device)

            mean_0_a = z1_flat[:784]
            mean_0_b = mu_left_z1

            mean_1_a = z2_flat[784:]
            mean_1_b = mu_right_z2

            z_hybrid_a = prior.get_full_latent([mean_0_a.unsqueeze(0), mean_1_a.unsqueeze(0)])
            z_hybrid_b = prior.get_full_latent([mean_0_a.unsqueeze(0), mean_1_b.unsqueeze(0)])
            z_hybrid_c = prior.get_full_latent([mean_0_b.unsqueeze(0), mean_1_a.unsqueeze(0)])

            z_hybrid_structural_a = z_hybrid_a.view(1, 4, 14, 28)
            z_hybrid_structural_b = z_hybrid_b.view(1, 4, 14, 28)
            z_hybrid_structural_c = z_hybrid_c.view(1, 4, 14, 28)

            img_hybrid_a = model.inverse(z_hybrid_structural_a)
            img_hybrid_b = model.inverse(z_hybrid_structural_b)
            img_hybrid_c = model.inverse(z_hybrid_structural_c)

        axes[row, 0].imshow(img1.squeeze().cpu(), cmap='gray')
        axes[row, 0].set_title(f"Image 1: {label1.tolist()}")
        axes[row, 0].axis('off')
        axes[row, 1].imshow(img2.squeeze().cpu(), cmap='gray')
        axes[row, 1].set_title(f"Image 2: {label2.tolist()}")
        axes[row, 1].axis('off')
        axes[row, 2].imshow(img_hybrid_a.squeeze().cpu(), cmap='gray')
        axes[row, 2].set_title("Hybrid: L1+R2")
        axes[row, 2].axis('off')
        axes[row, 3].imshow(img_hybrid_b.squeeze().cpu(), cmap='gray')
        axes[row, 3].set_title("Hybrid: L1+R2n")
        axes[row, 3].axis('off')
        axes[row, 4].imshow(img_hybrid_c.squeeze().cpu(), cmap='gray')
        axes[row, 4].set_title("Hybrid: L1n+R2")
        axes[row, 4].axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{config_str}_hybrid_reconstructions.png")
    plt.close(fig4)

# ==========================================
# 4. MAIN LOOP
# ==========================================

# Generate all combinations
keys, values = zip(*HYPERPARAMS.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

results = []

set_seed(42)

for config in combinations:
    # Construct ID and Path
    # ID Format: best_acc_1.0_hybrid_v2_Adam_0.5_0.2
    config_id = f"{config['TYPE']}_{config['SCALE']}_{config['MODEL']}_{config['PRIOR']}_{config['OPTIMIZER']}_{config['TRANSFORM']}_{config['DROPOUT']}"
    model_path = f"../experiments/models/GMM/{config_id}.pth"
    
    print(f"\nProcessing: {config_id}")
    
    if not os.path.exists(model_path):
        print(f"  [!] Checkpoint not found: {model_path}. Skipping.")
        continue

    # Load Model Class Dynamically
    try:
        module = importlib.import_module(config['MODEL'])
        GeneralFlow = getattr(module, 'GeneralFlow')
        
        # Initialize
        prior = SimpleSplitGMM(total_dim=1568, num_classes=10, num_attr=2, device=DEVICE, scale=config['SCALE'], fixed_means=True)
        model = GeneralFlow().to(DEVICE)
        
        # Load State
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        prior.load_state_dict(checkpoint['prior_state_dict'])
        prior.means = checkpoint['means']
        print(f"  Loaded epoch: {checkpoint['epoch']}")
        
    except Exception as e:
        print(f"  [!] Error loading model: {e}")
        continue

    # --- Compute Metrics ---
    print("  Evaluating Validation Set...")
    val_accs, val_losses = evaluate_model(model, prior, val_loader, DEVICE, n_runs=N_RUNS)
    val_acc_mean, val_acc_ci = get_confidence_interval(val_accs)
    val_loss_mean, val_loss_ci = get_confidence_interval(val_losses)
    
    print("  Evaluating Test Set...")
    test_accs, test_losses = evaluate_model(model, prior, test_loader, DEVICE, n_runs=N_RUNS)
    test_acc_mean, test_acc_ci = get_confidence_interval(test_accs)
    test_loss_mean, test_loss_ci = get_confidence_interval(test_losses)
    
    print(f"  -> Val Acc: {val_acc_mean:.4f} ± {val_acc_ci:.4f}")
    print(f"  -> Val Loss: {val_loss_mean:.4f} ± {val_loss_ci:.4f}")
    
    # --- Generate Plots ---
    print("  Generating Plots...")
    try:
        generate_plots(model, prior, DEVICE, config_id, PLOT_DIR)
    except Exception as e:
        print(f"  [!] Error generating plots: {e}")

    # --- Save Results ---
    result_entry = config.copy()
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
    results.append(result_entry)

# ==========================================
# 5. WRITE TO CSV
# ==========================================
if results:
    df = pd.DataFrame(results)
    
    # Reorder columns for readability
    cols = list(HYPERPARAMS.keys()) + [
        "val_acc_mean", "val_acc_ci", "val_loss_mean", "val_loss_ci",
        "test_acc_mean", "test_acc_ci", "test_loss_mean", "test_loss_ci"
    ]
    df = df[cols]
    
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nCompleted. Results saved to {RESULTS_CSV}")
else:
    print("\nNo valid configurations processed.")