import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib
import sys

sys.path.append(os.path.abspath(os.path.join('..', '..')))
from utils import set_seed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../priors')))
from CheckerboardIB import CheckerboardIB
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

BETAS = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

FIXED_CONFIG = {
    "SCALE": 0.0,
    "MODEL": "hybrid_v3_1x1_double",      
    "PRIOR": "CheckerboardIB",
    "OPTIMIZER": "Adam",
    "TRANSFORM": 0.5,
    "DROPOUT": 0.1,
    "TYPE": "best_loss",
    "FIXED_MEANS": False,
    "EPOCHS_WARMUP": 20,
    "STYLE_VARIANCE": 1.0
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOT_DIR = "interpolation_plots/"
os.makedirs(PLOT_DIR, exist_ok=True)

DIGIT_A_0, DIGIT_A_1 = 1, 8
DIGIT_B_0, DIGIT_B_1 = 2, 3
NUM_STEPS = 11
TEMP = 0.25

def unified_interpolation_plot(beta_data_list, save_dir):
    """
    Plots a unified interpolation grid.
    beta_data_list: list of tuples (beta, list_of_11_images)
    """
    if not beta_data_list:
        print("No valid models found to plot.")
        return

    num_betas = len(beta_data_list)
    fig, axes = plt.subplots(num_betas, NUM_STEPS, figsize=(22, 1.6 * num_betas))
    
    axes = np.atleast_2d(axes)
    
    alphas = np.linspace(0, 1, NUM_STEPS)

    for row, (beta, images) in enumerate(beta_data_list):
        for col, (alpha, img_gen) in enumerate(zip(alphas, images)):
            ax = axes[row, col]
            ax.imshow(img_gen, cmap='gray')
            ax.axis('off')
            
            if row == 0:
                ax.set_title(r"$\alpha=$" f"{alpha:.1f}", fontsize=24)
                
                
            if col == 0:
                ax.text(-0.25, 0.5, r"$\beta=$" f"{beta}", fontsize=24, rotation=90,
                        transform=ax.transAxes, va='center', ha='right')
                
    plt.tight_layout()
    save_path = os.path.join(save_dir, "interpolation_IB.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Unified plot saved successfully at: {save_path}")

collected_data = []
set_seed(42)

for beta in BETAS:
    config_id = (f"{FIXED_CONFIG['TYPE']}_{FIXED_CONFIG['SCALE']}_{FIXED_CONFIG['MODEL']}_"
                 f"{FIXED_CONFIG['PRIOR']}_{beta}_{FIXED_CONFIG['OPTIMIZER']}_"
                 f"{FIXED_CONFIG['TRANSFORM']}_{FIXED_CONFIG['DROPOUT']}_"
                 f"{FIXED_CONFIG['FIXED_MEANS']}_{FIXED_CONFIG['EPOCHS_WARMUP']}_"
                 f"{FIXED_CONFIG['STYLE_VARIANCE']}")
                 
    model_path = f"../experiments/models/IB/{config_id}.pth"
    
    print(f"\nProcessing Beta: {beta}")
    
    if not os.path.exists(model_path):
        print(f"  [!] Checkpoint not found: {model_path}. Skipping beta {beta}.")
        continue

    try:
        module = importlib.import_module(FIXED_CONFIG['MODEL'])
        GeneralFlow = getattr(module, 'GeneralFlow')
        
        prior_class = globals()[FIXED_CONFIG['PRIOR']]
        prior = prior_class(
            total_dim=1568, 
            arr_num_classes=[10,10], 
            beta=beta, 
            device=DEVICE, 
            scale=FIXED_CONFIG['SCALE'], 
            fixed_means=FIXED_CONFIG['FIXED_MEANS']
        )
        model = GeneralFlow().to(DEVICE)
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        prior.load_state_dict(checkpoint['prior_state_dict'])
        
        if 'means' in checkpoint and not getattr(prior, 'fixed_means', False):
            with torch.no_grad():
                for i in range(len(prior.means)):
                    prior.means[i].copy_(checkpoint['means'][i])

        print(f"  Loaded epoch: {checkpoint.get('epoch', 'Unknown')}")
        
        model.eval()
        prior.eval()
        
        images_for_beta = []
        with torch.no_grad():
            mean_a_0 = prior.means[0][DIGIT_A_0].unsqueeze(0).to(DEVICE)
            mean_a_1 = prior.means[1][DIGIT_A_1].unsqueeze(0).to(DEVICE)
            mean_b_0 = prior.means[0][DIGIT_B_0].unsqueeze(0).to(DEVICE)
            mean_b_1 = prior.means[1][DIGIT_B_1].unsqueeze(0).to(DEVICE)
            
            for alpha in np.linspace(0, 1, NUM_STEPS):
                z = prior.get_full_latent([(1-alpha) * mean_a_0 + alpha * mean_b_0, 
                                           (1-alpha) * mean_a_1 + alpha * mean_b_1])
                z = z + torch.randn_like(z) * TEMP
                z_structural = z.view(1, 4, 14, 28)
                img_gen = model.inverse(z_structural)
                
                images_for_beta.append(img_gen.squeeze().cpu().numpy())
        
        collected_data.append((beta, images_for_beta))
        
        del model
        del prior
        del checkpoint
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  [!] Error processing beta {beta}: {e}")

print("\nGenerating Unified Plot...")
unified_interpolation_plot(collected_data, PLOT_DIR)