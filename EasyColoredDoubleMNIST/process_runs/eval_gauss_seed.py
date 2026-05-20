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
sys.path.append(os.path.join(os.path.dirname(__file__), '../priors'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))

from GaussianPrior import GaussianPrior

HYPERPARAMS = {
    "MODEL": ["hybrid_v3_1x1_double"],                 
    "PRIOR": ["GaussianPrior"],               
    "OPTIMIZER": ["Adam"],
    "TRANSFORM": [0.5],
    "DROPOUT": [0.1],
    "TYPE": ["best_loss"],
    "VERSION": ["4_attr"]          
}

BATCH_SIZE = 128
EVAL_RUNS_PER_SEED = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_DIM = 3 * 28 * 56        

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
        for batch_X, batch_y in loader:
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

def evaluate_model(model, prior, loader, arr_num_classes, device, n_runs, 
                   comb_centroids, valid_combs, indep_centroids):
    """Computes Loss and both Classification types based on Z-distance."""
    model.eval()
    prior.eval()
    
    num_attr = len(arr_num_classes)
    losses = []
    
    comb_overall_accs = []
    comb_attr_accs = {k: [] for k in range(num_attr)}
    
    indep_overall_accs = []
    indep_attr_accs = {k: [] for k in range(num_attr)}

    for _ in range(n_runs):
        run_loss = 0.0
        run_total = 0
        
        run_comb_correct_all = 0
        run_comb_correct_attr = np.zeros(num_attr)
        run_indep_correct_all = 0
        run_indep_correct_attr = np.zeros(num_attr)
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                B = batch_X.size(0)
                
                batch_X = (batch_X * 255.0 + torch.rand_like(batch_X)) / 256.0 - 0.5

                z, sldj = model(batch_X)
                
                loss_output = prior.get_loss(z, sldj, batch_y)
                loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output
                run_loss += loss.item()

                z_flat = z.view(B, -1)

                comb_dists = torch.cdist(z_flat, comb_centroids)
                best_comb_indices = comb_dists.argmin(dim=1)
                best_comb_preds = valid_combs[best_comb_indices]
                
                run_comb_correct_all += (best_comb_preds == batch_y).all(dim=1).sum().item()
                for k in range(num_attr):
                    run_comb_correct_attr[k] += (best_comb_preds[:, k] == batch_y[:, k]).sum().item()

                best_indep_preds = torch.zeros((B, num_attr), device=device, dtype=torch.long)
                for k in range(num_attr):
                    indep_dists = torch.cdist(z_flat, indep_centroids[k])
                    best_indep_preds[:, k] = indep_dists.argmin(dim=1)
                    run_indep_correct_attr[k] += (best_indep_preds[:, k] == batch_y[:, k]).sum().item()
                    
                run_indep_correct_all += (best_indep_preds == batch_y).all(dim=1).sum().item()

                run_total += B
                
        losses.append(run_loss / len(loader))
        comb_overall_accs.append(run_comb_correct_all / run_total)
        indep_overall_accs.append(run_indep_correct_all / run_total)
        for k in range(num_attr):
            comb_attr_accs[k].append(run_comb_correct_attr[k] / run_total)
            indep_attr_accs[k].append(run_indep_correct_attr[k] / run_total)
            
    return comb_overall_accs, comb_attr_accs, indep_overall_accs, indep_attr_accs, losses

keys, values = zip(*HYPERPARAMS.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
loaded_datasets = {}
evaluated_models = set() 
all_results_list = [] 

set_seed(42)

for config in combinations:
    version = config['VERSION']
    prior_name = config['PRIOR']
    
    config_id = f"{config['TYPE']}_{config['MODEL']}_{prior_name}_{config['OPTIMIZER']}_{config['TRANSFORM']}_{config['DROPOUT']}"
    
    run_signature = (config_id, version)
    if run_signature in evaluated_models:
        continue
    evaluated_models.add(run_signature)
    
    all_model_paths = []
    
    original_model_path = f"../experiments/models/Gaussian/{version}/{config_id}.pth"
    if os.path.exists(original_model_path):
        all_model_paths.append(original_model_path)
    
    search_pattern = f"../experiments_seed/models/Gaussian/{version}/{config_id.rstrip('_')}_*.pth"
    all_model_paths.extend(glob.glob(search_pattern))
    
    print(f"\nProcessing: {config_id} [{version}]")
    if not all_model_paths:
        print(f"  [!] No checkpoints found.")
        continue
    else:
        print(f"  [+] Found {len(all_model_paths)} total runs.")

    if version not in loaded_datasets:
        print(f"  Loading Data for {version}...")
        
        file_name = 'easy_colored_double_mnist.npz' if version in ['1_attr', '2_attr'] else 'easy_colored_double_mnist_with_attributes.npz'
        data = np.load(f'../data/{file_name}')
        
        classes_map = {'1_attr': [10], '2_attr': [10, 10], '3_attr': [10, 10, 7], '4_attr': [10, 10, 7, 7]}
        arr_num_classes = classes_map[version]
        num_attr = len(arr_num_classes)
        
        X_train_t = torch.tensor(data['X_train'].transpose(0, 3, 1, 2), dtype=torch.float32)
        y_train_t = torch.tensor(data['y_train'][:, :num_attr], dtype=torch.long)
        
        X_val_t = torch.tensor(data['X_val'].transpose(0, 3, 1, 2), dtype=torch.float32)
        y_val_t = torch.tensor(data['y_val'][:, :num_attr], dtype=torch.long)
        
        X_test_t = torch.tensor(data['X_test'].transpose(0, 3, 1, 2), dtype=torch.float32)
        y_test_t = torch.tensor(data['y_test'][:, :num_attr], dtype=torch.long)
        
        loaded_datasets[version] = {
            'train_loader': DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=False),
            'val_loader': DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False),
            'test_loader': DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False),
            'arr_num_classes': arr_num_classes
        }
    
    current_data = loaded_datasets[version]
    num_attr = len(current_data['arr_num_classes'])

    seed_val_losses, seed_test_losses = [], []
    
    seed_val_comb_accs, seed_val_indep_accs = [], []
    seed_val_comb_attr_accs = {k: [] for k in range(num_attr)}
    seed_val_indep_attr_accs = {k: [] for k in range(num_attr)}

    seed_test_comb_accs, seed_test_indep_accs = [], []
    seed_test_comb_attr_accs = {k: [] for k in range(num_attr)}
    seed_test_indep_attr_accs = {k: [] for k in range(num_attr)}

    for model_path in all_model_paths:
        try:
            module = importlib.import_module(config['MODEL'])
            model = getattr(module, 'GeneralFlow')(dropout_p=config['DROPOUT']).to(DEVICE)
            
            prior = globals()[prior_name](
                device=DEVICE, 
                total_dim=TOTAL_DIM, 
                num_attr=len(current_data['arr_num_classes'])
            ).to(DEVICE)
            
            checkpoint = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            prior.load_state_dict(checkpoint['prior_state_dict'])
                 
        except Exception as e:
            print(f"  [!] Error loading {model_path}: {e}")
            continue

        comb_cen, valid_combs, indep_cen = compute_centroids(model, current_data['train_loader'], current_data['arr_num_classes'], DEVICE)

        v_comb_acc, v_comb_attr, v_indep_acc, v_indep_attr, v_losses = evaluate_model(
            model, prior, current_data['val_loader'], current_data['arr_num_classes'], 
            DEVICE, EVAL_RUNS_PER_SEED, comb_cen, valid_combs, indep_cen
        )
        
        seed_val_losses.append(np.mean(v_losses))
        seed_val_comb_accs.append(np.mean(v_comb_acc))
        seed_val_indep_accs.append(np.mean(v_indep_acc))
        for k in range(num_attr):
            seed_val_comb_attr_accs[k].append(np.mean(v_comb_attr[k]))
            seed_val_indep_attr_accs[k].append(np.mean(v_indep_attr[k]))

        t_comb_acc, t_comb_attr, t_indep_acc, t_indep_attr, t_losses = evaluate_model(
            model, prior, current_data['test_loader'], current_data['arr_num_classes'], 
            DEVICE, EVAL_RUNS_PER_SEED, comb_cen, valid_combs, indep_cen
        )
        
        seed_test_losses.append(np.mean(t_losses))
        seed_test_comb_accs.append(np.mean(t_comb_acc))
        seed_test_indep_accs.append(np.mean(t_indep_acc))
        for k in range(num_attr):
            seed_test_comb_attr_accs[k].append(np.mean(t_comb_attr[k]))
            seed_test_indep_attr_accs[k].append(np.mean(t_indep_attr[k]))

    if not seed_test_losses:
        continue

    print(f"  -> Test Loss: {np.nanmean(seed_test_losses):.4f}")
    print(f"  -> Test Acc (Comb) : {np.nanmean(seed_test_comb_accs):.4f}")
    print(f"  -> Test Acc (Indep): {np.nanmean(seed_test_indep_accs):.4f}")
    
    result_entry = config.copy()
    
    result_entry['val_loss_mean'], result_entry['val_loss_ci'] = get_confidence_interval(seed_val_losses)
    result_entry['test_loss_mean'], result_entry['test_loss_ci'] = get_confidence_interval(seed_test_losses)
    
    result_entry['val_acc_comb_mean'], result_entry['val_acc_comb_ci'] = get_confidence_interval(seed_val_comb_accs)
    result_entry['test_acc_comb_mean'], result_entry['test_acc_comb_ci'] = get_confidence_interval(seed_test_comb_accs)
    
    result_entry['val_acc_indep_mean'], result_entry['val_acc_indep_ci'] = get_confidence_interval(seed_val_indep_accs)
    result_entry['test_acc_indep_mean'], result_entry['test_acc_indep_ci'] = get_confidence_interval(seed_test_indep_accs)

    attr_names = ["L_Digit", "R_Digit", "L_Color", "R_Color"]
    for k in range(num_attr):
        col_name = attr_names[k] if k < 4 else f"Attr_{k}"
        
        result_entry[f"val_acc_{col_name}_comb_mean"], result_entry[f"val_acc_{col_name}_comb_ci"] = get_confidence_interval(seed_val_comb_attr_accs[k])
        result_entry[f"test_acc_{col_name}_comb_mean"], result_entry[f"test_acc_{col_name}_comb_ci"] = get_confidence_interval(seed_test_comb_attr_accs[k])
        
        result_entry[f"val_acc_{col_name}_indep_mean"], result_entry[f"val_acc_{col_name}_indep_ci"] = get_confidence_interval(seed_val_indep_attr_accs[k])
        result_entry[f"test_acc_{col_name}_indep_mean"], result_entry[f"test_acc_{col_name}_indep_ci"] = get_confidence_interval(seed_test_indep_attr_accs[k])

    all_results_list.append(result_entry)

if all_results_list:
    df_all_results = pd.DataFrame(all_results_list)
    unique_priors = df_all_results['PRIOR'].unique()
    
    PARAMS_TO_TRACK = ['VERSION'] 
    
    for prior_target in unique_priors:
        df_prior = df_all_results[df_all_results['PRIOR'] == prior_target]
        
        name_parts = ["Gaussian", prior_target]
        varying_parts = []
        
        for param in PARAMS_TO_TRACK:
            if param in df_prior.columns:
                unique_vals = df_prior[param].dropna().unique()
                
                if len(unique_vals) == 1:
                    val = unique_vals[0]
                    name_parts.append(str(val))
                elif len(unique_vals) > 1:
                    varying_parts.append(f"vary_{param.lower()}")
        
        file_suffix = "_".join(name_parts + varying_parts)
            
        csv_file_path = os.path.join(CSV_OUT_DIR, f"evaluation_results_{file_suffix}.csv")
        df_prior.to_csv(csv_file_path, index=False, float_format="%.6f")
        print(f"\n[+] Saved {prior_target} CSV to {csv_file_path}")
