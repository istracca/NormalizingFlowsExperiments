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
import argparse

sys.path.append(os.path.abspath(os.path.join('..', '..')))
from utils import set_seed
sys.path.append(os.path.join(os.path.dirname(__file__), '../priors'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))

from GaussianPrior import GaussianPrior

parser = argparse.ArgumentParser(description="Evaluate Conditional SingleMNIST Models Across Seeds")
parser.add_argument('--eval_runs_per_seed', type=int, default=20, help='Number of evaluation runs per seed for CI estimation')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
args = parser.parse_args()

HYPERPARAMS = {
    "MODEL": ["conditional_scale"],      
    "PRIOR": ["GaussianPrior"],
    "COND_DIM": [64],                 
    "OPTIMIZER": ["Adam"],
    "TRANSFORM": [0.5],
    "DROPOUT": [0.1],
    "TYPE": ["best_loss"]         
}

BATCH_SIZE = args.batch_size
EVAL_RUNS_PER_SEED = args.eval_runs_per_seed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_DIM = 784               
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

def evaluate_model(model, prior, loader, device, n_runs):
    """Computes Loss and Classification based on Z-distance to origin for SingleMNIST."""
    model.eval()
    prior.eval()
    
    losses = []
    accs = []

    all_comb_onehots = []
    for c in range(NUM_CLASSES):
        onehot = torch.nn.functional.one_hot(torch.tensor([c]), num_classes=NUM_CLASSES).float().to(device)
        all_comb_onehots.append(onehot)

    for _ in range(n_runs):
        run_loss = 0.0
        run_correct = 0
        run_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                B = batch_X.size(0)
                
                batch_X = (batch_X * 255.0 + torch.rand_like(batch_X)) / 256.0 - 0.5

                batch_y_onehot = torch.nn.functional.one_hot(batch_y, num_classes=NUM_CLASSES).float()

                z_true, sldj_true = model(batch_X, batch_y_onehot)
                loss_output = prior.get_loss(z_true, sldj_true, batch_y)
                loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output
                run_loss += loss.item()

                min_nll = torch.full((B,), float('inf'), device=device)
                best_preds = torch.zeros((B,), device=device, dtype=torch.long)

                for c in range(NUM_CLASSES):
                    cond_onehot = all_comb_onehots[c].expand(B, -1)
                    
                    z_test, sldj_test = model(batch_X, cond_onehot)
                    
                    z_flat = z_test.view(B, -1)
                    prior_nll = 0.5 * (z_flat ** 2).sum(dim=1) + 0.5 * z_flat.shape[1] * np.log(2 * np.pi)
                    
                    total_nll = prior_nll - sldj_test
                    
                    update_mask = total_nll < min_nll
                    min_nll[update_mask] = total_nll[update_mask]
                    best_preds[update_mask] = c

                run_correct += (best_preds == batch_y).sum().item()
                run_total += B
                
        losses.append(run_loss / len(loader))
        accs.append(run_correct / run_total)
            
    return accs, losses

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
    config_id = f"{config['TYPE']}_{config['MODEL']}_{config['PRIOR']}_{config['OPTIMIZER']}_{config['TRANSFORM']}_{config['DROPOUT']}"
    
    if config_id in evaluated_models:
        continue
    evaluated_models.add(config_id)
    
    all_model_paths = []
    
    original_model_path = f"../experiments/models/Conditional/{config_id}.pth"
    fallback_model_path = f"../experiments/models/Conditional/{config['TYPE']}_{config['MODEL']}_{config['PRIOR']}_{config['OPTIMIZER']}_{config['TRANSFORM']}_{config['DROPOUT']}.pth"

    if os.path.exists(original_model_path):
        all_model_paths.append(original_model_path)
    elif os.path.exists(fallback_model_path):
        all_model_paths.append(fallback_model_path)
    
    search_pattern = f"../experiments_seed/models/Conditional/{config_id.rstrip('_')}_*.pth"
    all_model_paths.extend(glob.glob(search_pattern))
    
    print(f"\nProcessing: {config_id}")
    if not all_model_paths:
        print(f"  [!] No checkpoints found.")
        continue
    else:
        print(f"  [+] Found {len(all_model_paths)} total runs.")

    seed_val_accs, seed_val_losses = [], []
    seed_test_accs, seed_test_losses = [], []

    for model_path in all_model_paths:
        try:
            module = importlib.import_module(config['MODEL'])
            model = getattr(module, 'GeneralFlow')(
                dropout_p=config['DROPOUT'], 
                num_classes=NUM_CLASSES, 
                cond_dim=config['COND_DIM']
            ).to(DEVICE)
            
            prior = globals()[config['PRIOR']](
                device=DEVICE, 
                total_dim=TOTAL_DIM, 
                num_classes=NUM_CLASSES
            ).to(DEVICE)
            
            checkpoint = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'prior_state_dict' in checkpoint:
                prior.load_state_dict(checkpoint['prior_state_dict'])
                 
        except Exception as e:
            print(f"  [!] Error loading {model_path}: {e}")
            continue

        v_acc_all, v_losses = evaluate_model(model, prior, val_loader, DEVICE, EVAL_RUNS_PER_SEED)
        seed_val_accs.append(np.mean(v_acc_all))
        seed_val_losses.append(np.mean(v_losses))

        t_acc_all, t_losses = evaluate_model(model, prior, test_loader, DEVICE, EVAL_RUNS_PER_SEED)
        seed_test_accs.append(np.mean(t_acc_all))
        seed_test_losses.append(np.mean(t_losses))

    if not seed_test_accs:
        continue

    val_metrics = {
        "val_acc": get_confidence_interval(seed_val_accs), 
        "val_loss": get_confidence_interval(seed_val_losses)
    }
    test_metrics = {
        "test_acc": get_confidence_interval(seed_test_accs), 
        "test_loss": get_confidence_interval(seed_test_losses)
    }
    
    print(f"  -> Aggregate Test Loss: {test_metrics['test_loss'][0]:.4f} ± {test_metrics['test_loss'][1]:.4f}")
    print(f"  -> Aggregate Test Acc : {test_metrics['test_acc'][0]:.4f} ± {test_metrics['test_acc'][1]:.4f}")
    
    result_entry = config.copy()
    for prefix, metrics in [("val", val_metrics), ("test", test_metrics)]:
        for key, (mean, ci) in metrics.items():
            result_entry[f"{key}_mean"] = mean
            result_entry[f"{key}_ci"] = ci

    all_results_list.append(result_entry)

if all_results_list:
    df_all_results = pd.DataFrame(all_results_list)
    unique_priors = df_all_results['PRIOR'].unique()
    
    PARAMS_TO_TRACK = ['MODEL'] 
    
    for prior_target in unique_priors:
        df_prior = df_all_results[df_all_results['PRIOR'] == prior_target]
        
        name_parts = ["Conditional_SingleMNIST", prior_target]
        varying_parts = []
        
        for param in PARAMS_TO_TRACK:
            if param in df_prior.columns:
                unique_vals = df_prior[param].dropna().unique()
                if len(unique_vals) > 1:
                    varying_parts.append(f"vary_{param.lower()}")
        
        file_suffix = "_".join(name_parts + varying_parts)
            
        csv_file_path = os.path.join(CSV_OUT_DIR, f"evaluation_results_{file_suffix}.csv")
        df_prior.to_csv(csv_file_path, index=False, float_format="%.6f")
        print(f"\n[+] Saved {prior_target} CSV to {csv_file_path}")
