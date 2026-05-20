import os
import sys
import torch
import random
import importlib
import math
import itertools
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

sys.path.append(os.path.abspath(os.path.join('..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../priors')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from utils import set_seed
from CheckerboardGMM import CheckerboardGMM

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='4_attr', help='Version of the dataset')
parser.add_argument('--scale', type=float, default=1.0, help='Scale parameter')
parser.add_argument('--disc_model', type=str, default='disc_v3_double', help='Discriminator model architecture')
parser.add_argument('--disc_path', type=str, default='../experiments/models/Disc/4_attr/best_loss_disc_v3_double_Adam_0.5_0.1.pth', help='Path to discriminator weights')
args = parser.parse_args()

VERSION = args.version
SCALE = args.scale
SEED = 42

if VERSION == '1_attr':
    ARR_NUM_CLASSES = [10]
elif VERSION == '2_attr':
    ARR_NUM_CLASSES = [10, 10]
elif VERSION == '3_attr':
    ARR_NUM_CLASSES = [10, 10, 7]
elif VERSION == '4_attr':
    ARR_NUM_CLASSES = [10, 10, 7, 7]
else:
    raise ValueError(f"Unknown VERSION: {VERSION}")

NUM_ATTR = len(ARR_NUM_CLASSES)

MODEL_CONFIG = {
    "MODEL": "hybrid_v3_1x1_double", 
    "PATH": f"../experiments/models/GMM/{VERSION}/best_loss_{SCALE}_hybrid_v3_1x1_double_CheckerboardGMM_Adam_0.5_0.1.pth",
    "PRIOR": "CheckerboardGMM",
    "DROPOUT": 0.1,
    "TOTAL_DIM": 4704
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_CSV = f"csv/GMM_{SCALE}_{VERSION}/class_cons_GMM.csv"
PLOT_DIR = f"plots/GMM_{SCALE}_{VERSION}/"
SAMPLES_PER_CONCEPT = 1000 
TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8]

rng = np.random.default_rng(SEED)
SEEDS = ["original"] + rng.integers(0, 1000000, size=4).tolist()

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def get_confidence_interval(data_array, confidence=0.95):
    """Calculates mean and error margin for CI."""
    data_array = np.array(data_array, dtype=float)
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

def parse_predictions(preds_raw):
    """Parses raw predictions from the GMM prior into a standardized tensor format."""
    if isinstance(preds_raw, list):
        return torch.stack(preds_raw, dim=1)
    elif isinstance(preds_raw, tuple):
        return torch.stack(preds_raw[0], dim=1) if isinstance(preds_raw[0], list) else preds_raw[0]
    return preds_raw

def evaluate_integrated(model, prior, disc_model, device, is_original=False):
    model.eval()
    prior.eval()
    disc_model.eval()
    
    temp_aggregates = {}
    
    all_combinations = list(itertools.product(*[range(n) for n in ARR_NUM_CLASSES]))
    total_combinations = len(all_combinations)
    
    for temp in TEMPERATURES:
        my_samples_per_concept = 1 if temp == 0.0 else SAMPLES_PER_CONCEPT
        print(f"  -> Evaluating Temperature: {temp} ({total_combinations} combinations)")
        
        temp_consistencies = [] 
        temp_disc_accuracies = []
        
        with torch.no_grad():
            for combo_idx, combo in enumerate(all_combinations):
                
                z_parts = []
                for k, c in enumerate(combo):
                    mean_k = prior.means[k][c].to(device).unsqueeze(0).expand(my_samples_per_concept, -1)
                    z_parts.append(mean_k)
                
                z = prior.get_full_latent(z_parts)
                if temp > 0:
                    z = z + torch.randn_like(z) * temp
                    
                z_structural = z.view(my_samples_per_concept, 12, 14, 28)
                img_gen = model.inverse(z_structural)
                
                z_post, _ = model(img_gen)
                z_post_flat = z_post.view(my_samples_per_concept, -1)
                
                preds_post_raw, _ = prior.classify(z_post_flat)
                preds_post = parse_predictions(preds_post_raw)
                
                disc_logits = disc_model(img_gen)
                disc_preds = torch.stack([logit.argmax(dim=1) for logit in disc_logits], dim=1)

                correct_gen = torch.ones(my_samples_per_concept, dtype=torch.bool, device=device)
                correct_disc = torch.ones(my_samples_per_concept, dtype=torch.bool, device=device)

                for k, c in enumerate(combo):
                    correct_gen &= (preds_post[:, k] == c)
                    correct_disc &= (disc_preds[:, k] == c)
                
                temp_consistencies.append(correct_gen.float().mean().item())
                temp_disc_accuracies.append(correct_disc.float().mean().item())
        
        temp_aggregates[temp] = {
            'consistency_ratio': float(np.mean(temp_consistencies)),
            'disc_accuracy': float(np.mean(temp_disc_accuracies))
        }
            
    return temp_aggregates


if __name__ == "__main__":
    set_seed(SEED)
    
    print("Loading Architectures...")
    model_module = importlib.import_module(MODEL_CONFIG['MODEL'])
    GeneralFlow = getattr(model_module, 'GeneralFlow')
    prior_class = globals()[MODEL_CONFIG['PRIOR']]
    
    disc_module = importlib.import_module(args.disc_model)
    PseudoResNet = getattr(disc_module, 'PseudoResNet')
    
    disc_model = PseudoResNet(arr_num_classes=ARR_NUM_CLASSES, in_channels=3, dropout_p=0.1, device=DEVICE).to(DEVICE)
    try:
        disc_checkpoint = torch.load(args.disc_path, map_location=DEVICE)
        disc_model.load_state_dict(disc_checkpoint['model_state_dict'])
        print(f"Successfully loaded discriminator from {args.disc_path}")
    except FileNotFoundError:
        print(f"Error: Discriminator checkpoint not found at {args.disc_path}.")
        sys.exit(1)

    all_results = []
    print(f"Configuration: {VERSION} | Total Attributes: {NUM_ATTR}")
    
    for seed in SEEDS:
        print(f"\n[{'='*40}]")
        print(f"Processing Model Seed: {seed}")
        print(f"[{'='*40}]")
        
        if seed == 'original':
            model_path = MODEL_CONFIG['PATH']
            is_original = True
        else:
            base_path = MODEL_CONFIG['PATH'].replace('../experiments/', '../experiments_seed/')
            model_path = base_path.replace('.pth', f'_{seed}.pth')
            is_original = False
            
        prior = prior_class(total_dim=MODEL_CONFIG['TOTAL_DIM'], arr_num_classes=ARR_NUM_CLASSES, device=DEVICE, scale=SCALE, fixed_means=True)
        model = GeneralFlow(dropout_p=MODEL_CONFIG['DROPOUT']).to(DEVICE)
        
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            prior.load_state_dict(checkpoint['prior_state_dict'])
            prior.means = checkpoint['means']
        except FileNotFoundError:
            print(f"Warning: Checkpoint not found at {model_path}. Skipping seed {seed}.")
            continue
            
        temp_results = evaluate_integrated(model, prior, disc_model, DEVICE, is_original=is_original)
        
        for temp, metrics in temp_results.items():
            all_results.append({
                'temperature': temp,
                'seed': seed,
                'consistency_ratio': metrics['consistency_ratio'],
                'disc_accuracy': metrics['disc_accuracy']
            })

    df_raw = pd.DataFrame(all_results)
    final_rows = []
    
    for temp in TEMPERATURES:
        temp_cons_data = df_raw[df_raw['temperature'] == temp]['consistency_ratio'].values
        temp_disc_data = df_raw[df_raw['temperature'] == temp]['disc_accuracy'].values
        
        for _, row in df_raw[df_raw['temperature'] == temp].iterrows():
            final_rows.append({
                'temperature': temp,
                'seed': row['seed'],
                'consistency_ratio': row['consistency_ratio'],
                'ci_margin': np.nan,
                'disc_accuracy': row['disc_accuracy'],
                'disc_ci_margin': np.nan
            })
            
        cons_mean, cons_h = get_confidence_interval(temp_cons_data)
        disc_mean, disc_h = get_confidence_interval(temp_disc_data)
        
        final_rows.append({
            'temperature': temp,
            'seed': 'aggregate',
            'consistency_ratio': cons_mean,
            'ci_margin': cons_h,
            'disc_accuracy': disc_mean,
            'disc_ci_margin': disc_h
        })
        
    df_final = pd.DataFrame(final_rows)
    df_final.to_csv(RESULTS_CSV, index=False)
    print(f"\nEvaluation complete. Aggregated metrics with CIs saved to '{RESULTS_CSV}'.")