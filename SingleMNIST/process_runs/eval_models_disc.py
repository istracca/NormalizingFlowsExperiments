import os
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
import itertools
import importlib
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append('../..')
from utils import set_seed

# ==========================================
# 1. CONFIGURATION GRID
# ==========================================
# Define the combinations you want to test here
HYPERPARAMS = {
    "MODEL": ["disc_v3"],      
    "OPTIMIZER": ["Adam"],
    "TRANSFORM": [0.0, 0.25, 0.5],
    "DROPOUT": [0.0, 0.1, 0.2],
    "TYPE": ["best_loss", "best_acc"]         
}

# Settings
BATCH_SIZE = 128
N_RUNS = 50          # Number of passes for CI calculation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_CSV = "disc_experiment_results.csv"


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

def evaluate_model(model, loader, device, n_runs=10):
    """
    Computes Accuracy and Loss using the specified logic.
    Returns arrays of results (one per run) for CI calculation.
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
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
                logits = model(batch_X)
            
                
                # --- Loss Calculation (User Logic) ---
                loss = criterion(logits, batch_y)
                run_loss += loss.item()
                
                # --- Accuracy Check ---
                preds = torch.argmax(logits, dim=1)

                run_correct += (preds == batch_y).sum().item()
                run_total += batch_y.size(0)
        
        # Aggregate for this run
        avg_loss = run_loss / len(loader) # Average batch loss
        avg_acc = run_correct / run_total
        
        losses.append(avg_loss)
        accuracies.append(avg_acc)
        
    return np.array(accuracies), np.array(losses)


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
    config_id = f"{config['TYPE']}_{config['MODEL']}_{config['OPTIMIZER']}_{config['TRANSFORM']}_{config['DROPOUT']}"
    model_path = f"../experiments/models/Disc/{config_id}.pth"
    
    print(f"\nProcessing: {config_id}")
    
    if not os.path.exists(model_path):
        print(f"  [!] Checkpoint not found: {model_path}. Skipping.")
        continue

    # Load Model Class Dynamically
    try:
        module = importlib.import_module(config['MODEL'])
        PseudoResNet = getattr(module, 'PseudoResNet')
        
        # Initialize
        model = PseudoResNet(num_classes = 10, dropout_p=config['DROPOUT']).to(DEVICE)
        
        # Load State
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded epoch: {checkpoint['epoch']}")
        
    except Exception as e:
        print(f"  [!] Error loading model: {e}")
        continue

    # --- Compute Metrics ---
    print("  Evaluating Validation Set...")
    val_accs, val_losses = evaluate_model(model, val_loader, DEVICE, n_runs=N_RUNS)
    val_acc_mean, val_acc_ci = get_confidence_interval(val_accs)
    val_loss_mean, val_loss_ci = get_confidence_interval(val_losses)
    
    print("  Evaluating Test Set...")
    test_accs, test_losses = evaluate_model(model, test_loader, DEVICE, n_runs=N_RUNS)
    test_acc_mean, test_acc_ci = get_confidence_interval(test_accs)
    test_loss_mean, test_loss_ci = get_confidence_interval(test_losses)
    
    print(f"  -> Val Acc: {val_acc_mean:.4f} ± {val_acc_ci:.4f}")
    print(f"  -> Val Loss: {val_loss_mean:.4f} ± {val_loss_ci:.4f}")
    

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