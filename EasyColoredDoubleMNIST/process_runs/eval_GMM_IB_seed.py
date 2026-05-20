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
import glob

sys.path.append(os.path.abspath(os.path.join('..', '..')))
from utils import set_seed
sys.path.append(os.path.join(os.path.dirname(__file__), '../priors'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))

from SimpleSplitGMM import SimpleSplitGMM
from CheckerboardGMM import CheckerboardGMM
from SimpleSplitIB import SimpleSplitIB
from CheckerboardIB import CheckerboardIB

HYPERPARAMS = {
    "SCALE": [1.0,2.0,3.0,4.0,5.0],              
    "MODEL": ["hybrid_v3_1x1_double"],      
    "PRIOR": ["CheckerboardGMM"],
    "BETA": [0.5],                 
    "OPTIMIZER": ["Adam"],
    "TRANSFORM": [0.5],
    "DROPOUT": [0.1],
    "TYPE": ["best_loss"],
    "VERSION": ["4_attr"], 
    "FIXED_MEANS": [False],          
    "EPOCHS_WARMUP": [20]          
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

def evaluate_model(model, prior, loader, device, n_runs):
    """Computes Loss and Per-Attribute Accuracy."""
    model.eval()
    prior.eval()
    
    num_attr = len(prior.means)
    losses, gen_losses, cls_losses = [], [], []
    attr_accs = {k: [] for k in range(num_attr)}
    overall_accs = []
    
    is_ib_prior = 'IB' in prior.__class__.__name__
    
    for _ in range(n_runs):
        run_loss, run_gen_loss, run_cls_loss = 0.0, 0.0, 0.0
        correct_per_attr = np.zeros(num_attr)
        run_correct_all, run_total = 0, 0
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                batch_X = (batch_X * 255.0 + torch.rand_like(batch_X)) / 256.0
                batch_X = batch_X - 0.5

                z, sldj = model(batch_X)
                loss_output = prior.get_loss(z, sldj, batch_y)
                
                if isinstance(loss_output, tuple):
                    loss, gen_loss, cls_loss = loss_output
                    run_loss += loss.item()
                    run_gen_loss += gen_loss.item()
                    run_cls_loss += cls_loss.item()
                else:
                    loss = loss_output
                    run_loss += loss.item()
                
                z_flat = z.view(z.size(0), -1)
                preds, _ = prior.classify(z_flat)
                
                if isinstance(preds, list):
                    preds = torch.stack(preds, dim=1)
                elif isinstance(preds, tuple): 
                    preds = torch.stack(preds[0], dim=1) if isinstance(preds[0], list) else preds[0]
                
                if preds.dim() == 1:
                    preds = preds.unsqueeze(1)
                
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

keys, values = zip(*HYPERPARAMS.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
loaded_datasets = {}
evaluated_models = set() 
all_results_list = [] 

set_seed(42)

for config in combinations:
    version = config['VERSION']
    prior_name = config['PRIOR']
    is_ib = 'IB' in prior_name
    folder_type = "IB" if is_ib else "GMM"
    
    if is_ib:
        config_id = f"{config['TYPE']}_{config['SCALE']}_{config['MODEL']}_{prior_name}_{config['BETA']}_{config['OPTIMIZER']}_{config['TRANSFORM']}_{config['DROPOUT']}_{config['FIXED_MEANS']}_{config['EPOCHS_WARMUP']}_"
    else:
        config_id = f"{config['TYPE']}_{config['SCALE']}_{config['MODEL']}_{prior_name}_{config['OPTIMIZER']}_{config['TRANSFORM']}_{config['DROPOUT']}"
        config['BETA'] = 'N/A'
        config['FIXED_MEANS'] = 'N/A'
        config['EPOCHS_WARMUP'] = 'N/A'
    
    run_signature = (config_id, version)
    if run_signature in evaluated_models:
        continue
    evaluated_models.add(run_signature)
    
    all_model_paths = []
    
    original_model_path = f"../experiments/models/{folder_type}/{version}/{config_id}.pth"
    if os.path.exists(original_model_path):
        all_model_paths.append(original_model_path)
    
    search_pattern = f"../experiments_seed/models/{folder_type}/{version}/{config_id.rstrip('_')}_*.pth"
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
        
        X_val_t = torch.tensor(data['X_val'].transpose(0, 3, 1, 2), dtype=torch.float32)
        y_val_t = torch.tensor(data['y_val'][:, :num_attr], dtype=torch.long)
        
        X_test_t = torch.tensor(data['X_test'].transpose(0, 3, 1, 2), dtype=torch.float32)
        y_test_t = torch.tensor(data['y_test'][:, :num_attr], dtype=torch.long)
        
        loaded_datasets[version] = {
            'val_loader': DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False),
            'test_loader': DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False),
            'arr_num_classes': arr_num_classes
        }
    
    current_data = loaded_datasets[version]
    num_attr = len(current_data['arr_num_classes'])

    seed_val_accs, seed_val_losses, seed_val_gen_losses, seed_val_cls_losses = [], [], [], []
    seed_val_attr_accs = {k: [] for k in range(num_attr)}
    
    seed_test_accs, seed_test_losses, seed_test_gen_losses, seed_test_cls_losses = [], [], [], []
    seed_test_attr_accs = {k: [] for k in range(num_attr)}

    for model_path in all_model_paths:
        try:
            module = importlib.import_module(config['MODEL'])
            model = getattr(module, 'GeneralFlow')(dropout_p=config['DROPOUT']).to(DEVICE)
            
            prior_args = {
                'total_dim': TOTAL_DIM, 'arr_num_classes': current_data['arr_num_classes'], 
                'device': DEVICE, 'scale': config['SCALE'], 'fixed_means': config['FIXED_MEANS'] if is_ib else False
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

        v_acc_all, v_acc_attr, v_losses, v_gen, v_cls = evaluate_model(model, prior, current_data['val_loader'], DEVICE, EVAL_RUNS_PER_SEED)
        seed_val_accs.append(np.mean(v_acc_all))
        seed_val_losses.append(np.mean(v_losses))
        seed_val_gen_losses.append(np.mean(v_gen))
        seed_val_cls_losses.append(np.mean(v_cls))
        for k in range(num_attr): seed_val_attr_accs[k].append(np.mean(v_acc_attr[k]))

        t_acc_all, t_acc_attr, t_losses, t_gen, t_cls = evaluate_model(model, prior, current_data['test_loader'], DEVICE, EVAL_RUNS_PER_SEED)
        seed_test_accs.append(np.mean(t_acc_all))
        seed_test_losses.append(np.mean(t_losses))
        seed_test_gen_losses.append(np.mean(t_gen))
        seed_test_cls_losses.append(np.mean(t_cls))
        for k in range(num_attr): seed_test_attr_accs[k].append(np.mean(t_acc_attr[k]))

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
    
    print(f"  -> Aggregate Val Acc: {val_metrics['val_acc'][0]:.4f} ± {val_metrics['val_acc'][1]:.4f}")
    
    result_entry = config.copy()
    for prefix, metrics in [("val", val_metrics), ("test", test_metrics)]:
        for key, (mean, ci) in metrics.items():
            result_entry[f"{key}_mean"] = mean
            result_entry[f"{key}_ci"] = ci
            
    attr_names = ["L_Digit", "R_Digit", "L_Color", "R_Color"]
    for k in range(num_attr):
        col_name = attr_names[k] if k < 4 else f"Attr_{k}"
        result_entry[f"val_acc_{col_name}_mean"], result_entry[f"val_acc_{col_name}_ci"] = get_confidence_interval(seed_val_attr_accs[k])
        result_entry[f"test_acc_{col_name}_mean"], result_entry[f"test_acc_{col_name}_ci"] = get_confidence_interval(seed_test_attr_accs[k])

    all_results_list.append(result_entry)

if all_results_list:
    df_all_results = pd.DataFrame(all_results_list)
    unique_priors = df_all_results['PRIOR'].unique()
    
    PARAMS_TO_TRACK = ['VERSION', 'SCALE', 'BETA'] 
    
    for prior_target in unique_priors:
        df_prior = df_all_results[df_all_results['PRIOR'] == prior_target]
        
        name_parts = [prior_target]
        varying_parts = []
        
        for param in PARAMS_TO_TRACK:
            if param in df_prior.columns:
                unique_vals = df_prior[param].dropna().unique()
                
                unique_vals = [v for v in unique_vals if str(v) != 'N/A']
                
                if len(unique_vals) == 1:
                    val = unique_vals[0]
                    if param == 'VERSION':
                        name_parts.append(str(val))
                    else:
                        name_parts.append(f"{param.lower()}_{val}")
                elif len(unique_vals) > 1:
                    varying_parts.append(f"vary_{param.lower()}")
        
        file_suffix = "_".join(name_parts + varying_parts)
            
        csv_file_path = os.path.join(CSV_OUT_DIR, f"evaluation_results_{file_suffix}.csv")
        df_prior.to_csv(csv_file_path, index=False, float_format="%.6f")
        print(f"\n[+] Saved {prior_target} CSV to {csv_file_path}")
        
        latex_str = "\\begin{table}[h!]\n\\centering\n\\resizebox{\\textwidth}{!}{\n"
        latex_str += "\\begin{tabular}{l | c | c c c c}\n\\toprule\n"
        latex_str += "\\textbf{Version} & \\textbf{Overall Acc} & \\textbf{L\\_Digit} & \\textbf{R\\_Digit} & \\textbf{L\\_Color} & \\textbf{R\\_Color} \\\\\n\\midrule\n"
        
        versions_to_print = ["1_attr", "2_attr", "3_attr", "4_attr"]
        attr_names = ["L_Digit", "R_Digit", "L_Color", "R_Color"]
        
        for v in versions_to_print:
            v_data = df_prior[df_prior['VERSION'] == v]
            if v_data.empty:
                continue
                
            row = v_data.iloc[0]
            row_str = v.replace('_', '\\_')
            row_str += f" & ${row['test_acc_mean']:.4f} \\pm {row['test_acc_ci']:.4f}$"
            
            for attr in attr_names:
                test_mean_col = f"test_acc_{attr}_mean"
                test_ci_col = f"test_acc_{attr}_ci"
                
                if test_mean_col in row and pd.notna(row[test_mean_col]):
                    test_str = f"${row[test_mean_col]:.4f} \\pm {row[test_ci_col]:.4f}$"
                else:
                    test_str = "-"
                row_str += f" & {test_str}"
            
            latex_str += row_str + " \\\\\n"
            
        latex_str += "\\bottomrule\n\\end{tabular}\n}\n"
        
        caption_name = file_suffix.replace('_', '\\_')
        latex_str += "\\caption{Test Accuracy Overall and per Attribute for " + caption_name + " (Mean $\\pm$ 95\\% CI).}\n"
        latex_str += f"\\label{{tab:eval_{file_suffix.lower()}}}\n\\end{{table}}\n"
        
        latex_file_path = os.path.join(CSV_OUT_DIR, f"evaluation_table_{file_suffix}.tex")
        with open(latex_file_path, "w") as f:
            f.write(latex_str)
        print(f"[+] Saved {prior_target} LaTeX table to {latex_file_path}")