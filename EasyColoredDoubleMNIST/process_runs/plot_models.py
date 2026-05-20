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
from SimpleSplitIB import SimpleSplitIB
from CheckerboardIB import CheckerboardIB

HYPERPARAMS = {
    "SCALE": [0.0],              
    "MODEL": ["hybrid_v3_1x1_double"],      
    "PRIOR": ["CheckerboardIB"],
    "BETA": [0.01, 0.05, 0.1, 0.5, 1.0],                                          
    "OPTIMIZER": ["Adam"],
    "TRANSFORM": [0.5],
    "DROPOUT": [0.1],
    "TYPE": ["best_loss"],
    "VERSION": ["4_attr"], 
    "FIXED_MEANS": [False],                                   
    "EPOCHS_WARMUP": [20]                                   
}

BATCH_SIZE = 128
N_RUNS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOT_BASE_DIR = "plots/samples"
TOTAL_DIM = 3 * 28 * 56        

os.makedirs(PLOT_BASE_DIR, exist_ok=True)

def postprocess(img_tensor):
    img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(img, 0, 1)

def generate_plots(model, prior, device, config_str, save_dir, X_test_tensor, y_test_tensor):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    num_attr = len(prior.means)
    
    def build_latent(parts_list):
        current_len = len(parts_list)
        if current_len < num_attr:
            for k in range(current_len, num_attr):
                if isinstance(prior.means, nn.ParameterList):
                    default_mean = prior.means[k][0].view(-1)
                else:
                    default_mean = prior.means[k][0].view(-1)
                parts_list.append(default_mean.unsqueeze(0))
        return prior.get_full_latent(parts_list)

    print(f"   Generating Means Grid...")
    if isinstance(prior.means, nn.ParameterList):
        num_rows = prior.means[0].shape[0]
        num_cols = prior.means[1].shape[0]
    else:
        num_rows = prior.means[0].shape[0]
        num_cols = prior.means[1].shape[0]

    fig1, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    with torch.no_grad():
        for i in range(num_rows):
            for j in range(num_cols):
                mean_0 = prior.means[0][i].view(-1).unsqueeze(0)
                mean_1 = prior.means[1][j].view(-1).unsqueeze(0)
                z = build_latent([mean_0, mean_1])
                
                z_struct = z.view(1, 12, 14, 28)
                img_gen = model.inverse(z_struct)
                img_gen = img_gen + 0.5
                
                ax = axes[i, j] if num_rows > 1 and num_cols > 1 else axes
                if isinstance(ax, np.ndarray): 
                    if ax.ndim == 2: ax = ax[i,j]
                    else: ax = ax[max(i,j)]           
                
                ax.imshow(postprocess(img_gen))
                ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{config_str}_zero_temp.png")
    plt.close(fig1)

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
            img1_in = (img1 * 255.0 + torch.rand_like(img1)) / 256.0 - 0.5
            img2_in = (img2 * 255.0 + torch.rand_like(img2)) / 256.0 - 0.5
            
            z1, _ = model(img1_in)
            z2, _ = model(img2_in)
            z1_chunks = prior.get_parts(z1.squeeze(0).unsqueeze(0))
            z2_chunks = prior.get_parts(z2.squeeze(0).unsqueeze(0))
            
            mix_chunks = []
            for k in range(num_attr):
                if k % 2 == 0: mix_chunks.append(z1_chunks[k])
                else:          mix_chunks.append(z2_chunks[k])
            z_mix = prior.get_full_latent(mix_chunks)
            rec_mix = model.inverse(z_mix.view(1, 12, 14, 28)) + 0.5
            
            mix_gen_chunks = []
            for k in range(num_attr):
                if k % 2 == 0: 
                    mix_gen_chunks.append(z1_chunks[k])
                else:
                    target = lbl2[k]
                    mix_gen_chunks.append(prior.means[k][target].view(-1).unsqueeze(0))
            z_mix_gen = prior.get_full_latent(mix_gen_chunks)
            rec_mix_gen = model.inverse(z_mix_gen.view(1, 12, 14, 28)) + 0.5

            mix_gen_chunks_2 = []
            for k in range(num_attr):
                if k % 2 == 0: 
                    target = lbl1[k]
                    mix_gen_chunks_2.append(prior.means[k][target].view(-1).unsqueeze(0))
                else:
                    mix_gen_chunks_2.append(z2_chunks[k])
            z_mix_gen_2 = prior.get_full_latent(mix_gen_chunks_2)
            rec_mix_gen_2 = model.inverse(z_mix_gen_2.view(1, 12, 14, 28)) + 0.5

        axes[row, 0].imshow(postprocess(img1)); axes[row, 0].set_title(f"I1: {lbl1.cpu().numpy()}")
        axes[row, 1].imshow(postprocess(img2)); axes[row, 1].set_title(f"I2: {lbl2.cpu().numpy()}")
        axes[row, 2].imshow(postprocess(rec_mix)); axes[row, 2].set_title("Mix Latents")
        axes[row, 3].imshow(postprocess(rec_mix_gen)); axes[row, 3].set_title("L1 + Mean2")
        axes[row, 4].imshow(postprocess(rec_mix_gen_2)); axes[row, 4].set_title("Mean1 + R2")
        for ax in axes[row]: ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{config_str}_hybrid.png")
    plt.close(fig2)


keys, values = zip(*HYPERPARAMS.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
results_by_group = {}
loaded_datasets = {}

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
    
    model_path = f"../experiments/models/{folder_type}/{version}/{config_id}.pth"
    plot_dir = os.path.join(PLOT_BASE_DIR, prior_name, version)
    
    print(f"\nProcessing: {config_id} [{version}]")
    
    if not os.path.exists(model_path):
        print(f"  [!] Checkpoint not found: {model_path}")
        continue

    if version not in loaded_datasets:
        print(f"  Loading Data for {version}...")
        if version == '2_attr':
            data_path = '../data/easy_colored_double_mnist.npz'
            arr_num_classes = [10, 10]
        elif version == '4_attr':
            data_path = '../data/easy_colored_double_mnist_with_attributes.npz'
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

    try:
        module = importlib.import_module(config['MODEL'])
        GeneralFlow = getattr(module, 'GeneralFlow')
        model = GeneralFlow(dropout_p=config['DROPOUT']).to(DEVICE)
        
        prior_class = globals()[prior_name]
        
        prior_args = {
            'total_dim': TOTAL_DIM, 
            'arr_num_classes': arr_num_classes, 
            'device': DEVICE, 
            'scale': config['SCALE'], 
            'fixed_means': config['FIXED_MEANS'] if is_ib else False                            
        }
        if is_ib:
            prior_args['beta'] = config['BETA']
            
        prior = prior_class(**prior_args).to(DEVICE)
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        prior.load_state_dict(checkpoint['prior_state_dict'])
        
        if 'means' in checkpoint and not isinstance(prior.means, nn.ParameterList):
             prior.means = checkpoint['means']
             
        print(f"  Loaded epoch: {checkpoint.get('epoch', 'Unknown')}")
        
    except Exception as e:
        print(f"  [!] Error loading model architecture: {e}")
        continue

    print("  Generating Plots...")
    try:
        generate_plots(model, prior, DEVICE, config_id, plot_dir, 
                       current_data['X_test_tensor'], current_data['y_test_tensor'])
    except Exception as e:
        print(f"  [!] Plot warning: {e}")