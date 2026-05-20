import os
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
import itertools
import importlib
from torch.utils.data import DataLoader, TensorDataset
import glob
import sys

sys.path.append(os.path.abspath(os.path.join('..', '..')))
from utils import set_seed
sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))

HYPERPARAMS = {
    "MODEL": ["disc_v3"],                                                               
    "OPTIMIZER": ["Adam"],
    "TRANSFORM": [0.5],
    "DROPOUT": [0.1],
    "TYPE": ["best_loss"]
}

BATCH_SIZE = 128
EVAL_RUNS_PER_SEED = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10

CSV_OUT_DIR = "csv"
os.makedirs(CSV_OUT_DIR, exist_ok=True)

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

def evaluate_model(model, loader, device, n_runs):
    """Computes overall accuracy over multiple forward passes."""
    model.eval()
    overall_accs = []

    for _ in range(n_runs):
        run_correct = 0
        run_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                B = batch_X.size(0)
                
                batch_X = (batch_X * 255.0 + torch.rand_like(batch_X)) / 256.0 - 0.5

                logits = model(batch_X)
                
                if isinstance(logits, list):
                    preds = logits[0].argmax(dim=1)
                else:
                    preds = logits.argmax(dim=1)

                run_correct += (preds == batch_y).sum().item()
                run_total += B
                
        overall_accs.append(run_correct / run_total)
            
    return overall_accs

keys, values = zip(*HYPERPARAMS.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
evaluated_models = set() 
all_results_list = [] 

set_seed(42)

print("Loading SingleMNIST Data...")
data = np.load('../data/mnist_data.npz')

X_val_t = torch.tensor(data['X_val'].reshape(-1, 1, 28, 28), dtype=torch.float32)
y_val_t = torch.tensor(data['y_val'], dtype=torch.long)

X_test_t = torch.tensor(data['X_test'].reshape(-1, 1, 28, 28), dtype=torch.float32)
y_test_t = torch.tensor(data['y_test'], dtype=torch.long)

val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

for config in combinations:
    
    config_id = f"{config['TYPE']}_{config['MODEL']}_{config['OPTIMIZER']}_{config['TRANSFORM']}_{config['DROPOUT']}"
    
    if config_id in evaluated_models:
        continue
    evaluated_models.add(config_id)
    
    all_model_paths = []
    
    original_model_path = f"../experiments/models/Disc/{config_id}.pth"
    if os.path.exists(original_model_path):
        all_model_paths.append(original_model_path)
    
    search_pattern = f"../experiments_seed/models/Disc/{config_id.rstrip('_')}_*.pth"
    all_model_paths.extend(glob.glob(search_pattern))
    
    print(f"\nProcessing: {config_id}")
    if not all_model_paths:
        print(f"  [!] No checkpoints found.")
        continue
    else:
        print(f"  [+] Found {len(all_model_paths)} total runs.")

    seed_val_accs, seed_test_accs = [], []

    for model_path in all_model_paths:
        try:
            module = importlib.import_module(config['MODEL'])
            ModelClass = getattr(module, 'PseudoResNet', getattr(module, 'DiscNet', None))
            
            model = ModelClass(
                num_classes=NUM_CLASSES,
                dropout_p=config['DROPOUT']
            ).to(DEVICE)
            
            checkpoint = torch.load(model_path, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)                                      
                 
        except Exception as e:
            print(f"  [!] Error loading {model_path}: {e}")
            continue

        v_acc_all = evaluate_model(model, val_loader, DEVICE, EVAL_RUNS_PER_SEED)
        seed_val_accs.append(np.mean(v_acc_all))

        t_acc_all = evaluate_model(model, test_loader, DEVICE, EVAL_RUNS_PER_SEED)
        seed_test_accs.append(np.mean(t_acc_all))

    if not seed_test_accs:
        continue

    val_metrics = {
        "val_acc": get_confidence_interval(seed_val_accs)
    }
    test_metrics = {
        "test_acc": get_confidence_interval(seed_test_accs)
    }
    
    print(f"  -> Aggregate Test Acc : {test_metrics['test_acc'][0]:.4f} ± {test_metrics['test_acc'][1]:.4f}")
    
    result_entry = config.copy()
    for prefix, metrics in [("val", val_metrics), ("test", test_metrics)]:
        for key, (mean, ci) in metrics.items():
            result_entry[f"{key}_mean"] = mean
            result_entry[f"{key}_ci"] = ci

    all_results_list.append(result_entry)

if all_results_list:
    df_all_results = pd.DataFrame(all_results_list)
    unique_models = df_all_results['MODEL'].unique()
    
    
    for model_target in unique_models:
        df_model = df_all_results[df_all_results['MODEL'] == model_target]
        
        name_parts = ["Disc_SingleMNIST", model_target]
        file_suffix = "_".join(name_parts)
            
        csv_file_path = os.path.join(CSV_OUT_DIR, f"evaluation_results_{file_suffix}.csv")
        df_model.to_csv(csv_file_path, index=False, float_format="%.6f")
        print(f"\n[+] Saved {model_target} CSV to {csv_file_path}")