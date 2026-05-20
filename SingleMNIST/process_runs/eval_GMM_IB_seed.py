import os
import torch
import torch.nn as nn
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
sys.path.append(os.path.join(os.path.dirname(__file__), '../priors'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))

from GMM import GMM
from IB import IB

HYPERPARAMS = {
    "SCALE": [0.0],              
    "MODEL": ["hybrid_v3_1x1"],  
    "PRIOR": ["IB"],
    "BETA": [0.01,0.05,0.1,0.5,1.0],                
    "OPTIMIZER": ["Adam"],
    "TRANSFORM": [0.5],
    "DROPOUT": [0.1],
    "TYPE": ["best_loss"],
    "FIXED_MEANS": [False],          
    "EPOCHS_WARMUP": [20]          
}

BATCH_SIZE = 128
EVAL_RUNS_PER_SEED = 20
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
    """Computes Loss and Accuracy for Single Attribute Prior."""
    model.eval()
    prior.eval()
    
    losses, gen_losses, cls_losses = [], [], []
    overall_accs = []
    
    is_ib_prior = 'IB' in prior.__class__.__name__
    
    for _ in range(n_runs):
        run_loss, run_gen_loss, run_cls_loss = 0.0, 0.0, 0.0
        run_correct, run_total = 0, 0
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                batch_X = (batch_X * 255.0 + torch.rand_like(batch_X)) / 256.0 - 0.5

                z, sldj = model(batch_X)
                
                if is_ib_prior:
                    loss, gen_loss, cls_loss = prior.get_loss(z, sldj, batch_y)
                    run_loss += loss.item()
                    run_gen_loss += gen_loss.item()
                    run_cls_loss += cls_loss.item()
                else:
                    loss = prior.get_loss(z, sldj, batch_y)
                    run_loss += loss.item()
                
                z_flat = z.view(z.size(0), -1)
                preds = prior.classify(z_flat)
                
                if isinstance(preds, tuple):
                    preds = preds[0]
                    
                run_correct += (preds == batch_y).sum().item()
                run_total += batch_y.size(0)
        
        losses.append(run_loss / len(loader))
        overall_accs.append(run_correct / run_total)

        if is_ib_prior:
            gen_losses.append(run_gen_loss / len(loader))
            cls_losses.append(run_cls_loss / len(loader))
        else:
            gen_losses.append(np.nan)
            cls_losses.append(np.nan)
            
    return overall_accs, losses, gen_losses, cls_losses

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
    prior_name = config['PRIOR']
    is_ib = 'IB' in prior_name
    folder_type = "IB" if is_ib else "GMM"
    
    if is_ib:
        config_id = f"{config['TYPE']}_{config['SCALE']}_{config['MODEL']}_{prior_name}_{config['BETA']}_{config['OPTIMIZER']}_{config['TRANSFORM']}_{config['DROPOUT']}_{config['FIXED_MEANS']}_{config['EPOCHS_WARMUP']}"
    else:
        config_id = f"{config['TYPE']}_{config['SCALE']}_{config['MODEL']}_{prior_name}_{config['OPTIMIZER']}_{config['TRANSFORM']}_{config['DROPOUT']}"
        config['BETA'] = 'N/A'
        config['FIXED_MEANS'] = 'N/A'
        config['EPOCHS_WARMUP'] = 'N/A'
    
    if config_id in evaluated_models:
        continue
    evaluated_models.add(config_id)
    
    all_model_paths = []
    
    original_model_path = f"../experiments/models/{folder_type}/{config_id}.pth"
    if os.path.exists(original_model_path):
        all_model_paths.append(original_model_path)
    
    search_pattern = f"../experiments_seed/models/{folder_type}/{config_id}_*.pth"
    all_model_paths.extend(glob.glob(search_pattern))
    
    print(f"\nProcessing: {config_id}")
    if not all_model_paths:
        print(f"  [!] No checkpoints found.")
        continue
    else:
        print(f"  [+] Found {len(all_model_paths)} total runs.")

    seed_val_accs, seed_val_losses, seed_val_gen_losses, seed_val_cls_losses = [], [], [], []
    seed_test_accs, seed_test_losses, seed_test_gen_losses, seed_test_cls_losses = [], [], [], []

    for model_path in all_model_paths:
        try:
            module = importlib.import_module(config['MODEL'])
            model = getattr(module, 'GeneralFlow')(dropout_p=config['DROPOUT']).to(DEVICE)
            
            prior_args = {
                'total_dim': TOTAL_DIM, 
                'num_classes': NUM_CLASSES,
                'device': DEVICE, 
                'scale': config['SCALE'], 
                'fixed_means': config['FIXED_MEANS'] if is_ib else True
            }
            if is_ib: prior_args['beta'] = config['BETA']
                
            prior = globals()[prior_name](**prior_args).to(DEVICE)
            checkpoint = torch.load(model_path, map_location=DEVICE)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            prior.load_state_dict(checkpoint['prior_state_dict'])
            
            if 'means' in checkpoint and not isinstance(prior.means, nn.ParameterList):
                 prior.means = checkpoint['means']
                 
        except Exception as e:
            print(f"  [!] Error loading {model_path}: {e}")
            continue

        v_acc_all, v_losses, v_gen, v_cls = evaluate_model(model, prior, val_loader, DEVICE, EVAL_RUNS_PER_SEED)
        seed_val_accs.append(np.mean(v_acc_all))
        seed_val_losses.append(np.mean(v_losses))
        seed_val_gen_losses.append(np.mean(v_gen))
        seed_val_cls_losses.append(np.mean(v_cls))

        t_acc_all, t_losses, t_gen, t_cls = evaluate_model(model, prior, test_loader, DEVICE, EVAL_RUNS_PER_SEED)
        seed_test_accs.append(np.mean(t_acc_all))
        seed_test_losses.append(np.mean(t_losses))
        seed_test_gen_losses.append(np.mean(t_gen))
        seed_test_cls_losses.append(np.mean(t_cls))

    if not seed_test_accs:
        continue

    val_metrics = {
        "val_acc": get_confidence_interval(seed_val_accs), "val_loss": get_confidence_interval(seed_val_losses),
        "val_gen_loss": get_confidence_interval(seed_val_gen_losses), "val_cls_loss": get_confidence_interval(seed_val_cls_losses),
    }
    test_metrics = {
        "test_acc": get_confidence_interval(seed_test_accs), "test_loss": get_confidence_interval(seed_test_losses),
        "test_gen_loss": get_confidence_interval(seed_test_gen_losses), "test_cls_loss": get_confidence_interval(seed_test_cls_losses),
    }
    
    print(f"  -> Aggregate Test Acc: {test_metrics['test_acc'][0]:.4f} ± {test_metrics['test_acc'][1]:.4f}")
    
    result_entry = config.copy()
    for prefix, metrics in [("val", val_metrics), ("test", test_metrics)]:
        for key, (mean, ci) in metrics.items():
            result_entry[f"{key}_mean"] = mean
            result_entry[f"{key}_ci"] = ci

    all_results_list.append(result_entry)

if all_results_list:
    df_all_results = pd.DataFrame(all_results_list)
    unique_priors = df_all_results['PRIOR'].unique()
    
    PARAMS_TO_TRACK = ['SCALE', 'BETA'] 
    
    for prior_target in unique_priors:
        df_prior = df_all_results[df_all_results['PRIOR'] == prior_target]
        
        name_parts = ["SingleMNIST", prior_target]
        varying_parts = []
        
        for param in PARAMS_TO_TRACK:
            if param in df_prior.columns:
                unique_vals = df_prior[param].dropna().unique()
                unique_vals = [v for v in unique_vals if str(v) != 'N/A']
                
                if len(unique_vals) == 1:
                    val = unique_vals[0]
                    name_parts.append(f"{param.lower()}_{val}")
                elif len(unique_vals) > 1:
                    varying_parts.append(f"vary_{param.lower()}")
        
        file_suffix = "_".join(name_parts + varying_parts)
            
        csv_file_path = os.path.join(CSV_OUT_DIR, f"evaluation_results_{file_suffix}.csv")
        df_prior.to_csv(csv_file_path, index=False, float_format="%.6f")
        print(f"\n[+] Saved {prior_target} CSV to {csv_file_path}")