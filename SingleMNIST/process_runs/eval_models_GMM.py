import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import itertools
import importlib
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append('../..')
from utils import set_seed
sys.path.append('../priors')
from GaussianMixturePrior import GaussianMixturePrior

# ==========================================
# 1. CONFIGURATION GRID
# ==========================================
# Define the combinations you want to test here
HYPERPARAMS = {
    "SCALE": [1.0, 2.0, 3.0],              
    "MODEL": ["hybrid_v3_1x1"],      
    "OPTIMIZER": ["Adam"],
    "TRANSFORM": [0.0, 0.25, 0.5],
    "DROPOUT": [0.0, 0.1, 0.2],
    "TYPE": ["best_loss", "best_acc"]         
}

# Settings
BATCH_SIZE = 128
N_RUNS = 50          # Number of passes for CI calculation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_CSV = "v3_experiment_results.csv"
PLOT_DIR = "plots/samples"  # <--- UPDATED PATH

# Create plot directory if it doesn't exist
os.makedirs(PLOT_DIR, exist_ok=True)

# ==========================================
# 2. DATA LOADING
# ==========================================
print("Loading Data...")
data = np.load('../data/mnist_data.npz')
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

# Prepare Tensors
def prepare_loader(X, y, batch_size, shuffle=False):
    X_tensor = torch.tensor(X.reshape(-1, 1, 28, 28), dtype=torch.float32)
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

def evaluate_model(model, prior, loader, device, n_runs=10):
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
                preds = prior.classify(z_flat)
                if isinstance(preds, tuple):
                    preds = preds[0]
                run_correct += (preds == batch_y).sum().item()
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
    targets = list(range(10))
    temp = 0
    fig1, axes = plt.subplots(2, 5, figsize=(15, 6))
    with torch.no_grad():
        for idx, target in enumerate(targets):
            # Prior mean
            z = prior.means[target].unsqueeze(0).to(device)
            z_structural = z.view(1, 4, 14, 14) # Adjust shape based on model
            img_gen = model.inverse(z_structural)
            
            ax = axes[idx // 5, idx % 5]
            ax.imshow(img_gen.squeeze().cpu(), cmap='gray')
            ax.set_title(f"Target: {target}")
            ax.axis('off')
    plt.suptitle(f"Zero Temp Generation ({config_str})")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{config_str}_zero_temp.png")
    plt.close(fig1)

    # 2. Low Temperature (0.25) Samples
    temp = 0.25
    fig2, axes = plt.subplots(10, 5, figsize=(15, 30)) # 10 rows (digits), 5 cols (samples)
    with torch.no_grad():
        for target in targets:
            for i in range(5):
                z = prior.means[target].unsqueeze(0).to(device) + \
                    torch.randn(1, prior.means.shape[1]).to(device) * temp
                z_structural = z.view(1, 4, 14, 14)
                img_gen = model.inverse(z_structural)
                
                ax = axes[target, i]
                ax.imshow(img_gen.squeeze().cpu(), cmap='gray')
                if i == 2: ax.set_title(f"Digit {target}")
                ax.axis('off')
    plt.suptitle(f"Temp 0.25 Samples ({config_str})")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{config_str}_samples.png")
    plt.close(fig2)

    # 3. Interpolation (3 -> 9)
    digit_a, digit_b = 3, 9 
    num_steps = 11
    temp = 0.1
    fig3, axes = plt.subplots(1, num_steps, figsize=(20, 3))
    with torch.no_grad():
        mean_a = prior.means[digit_a].unsqueeze(0).to(device)
        mean_b = prior.means[digit_b].unsqueeze(0).to(device)
        for i, alpha in enumerate(np.linspace(0, 1, num_steps)):
            z = (1 - alpha) * mean_a + alpha * mean_b
            z += torch.randn(1, prior.means.shape[1]).to(device) * temp
            z_structural = z.view(1, 4, 14, 14)
            img_gen = model.inverse(z_structural)
            
            ax = axes[i]
            ax.imshow(img_gen.squeeze().cpu(), cmap='gray')
            ax.set_title(f"α={alpha:.2f}")
            ax.axis('off')
    plt.suptitle(f"Interpolation {digit_a}->{digit_b} ({config_str})")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{config_str}_interp.png")
    plt.close(fig3)

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
    config_id = f"{config['TYPE']}_{config['SCALE']}_{config['MODEL']}_{config['OPTIMIZER']}_{config['TRANSFORM']}_{config['DROPOUT']}"
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
        prior = GaussianMixturePrior(total_dim=784, num_classes=10, device=DEVICE, scale=config['SCALE'], fixed_means=True)
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