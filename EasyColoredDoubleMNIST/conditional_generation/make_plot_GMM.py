import os
import sys
import torch
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse

sys.path.append(os.path.abspath(os.path.join('..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../priors')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from utils import set_seed
from CheckerboardGMM import CheckerboardGMM

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='4_attr', choices=['2_attr', '4_attr'], 
                    help='Dataset version to evaluate')
args = parser.parse_args()

VERSION = args.version
SCALES = [1.0, 2.0, 3.0, 4.0, 5.0]
TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8]

SAMPLES_PER_TEMP = {
    0.0: 1,
    0.2: 3,
    0.4: 3,
    0.6: 3,
    0.8: 3
}

if VERSION == '2_attr':
    ARR_NUM_CLASSES = [10, 10]
    TARGETS = [(0, 1), (2, 3), (4, 5)]
elif VERSION == '4_attr':
    ARR_NUM_CLASSES = [10, 10, 7, 7]
    TARGETS = [(0, 1, 0, 1), (2, 3, 2, 3), (4, 5, 4, 5)]

NUM_ATTR = len(ARR_NUM_CLASSES)
TOTAL_DIM = 4704 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = f"samples_plots/GMM/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_and_plot_grid():
    set_seed(42)
    
    num_scales = len(SCALES)
    num_targets = len(TARGETS)
    width_ratios = [0.25] + [SAMPLES_PER_TEMP[t] for t in TEMPERATURES]
    total_cols_visual = sum(width_ratios)
    
    fig = plt.figure(figsize=(total_cols_visual * 2.0, num_scales * num_targets * 1 + 1.5))
    fig.suptitle(f"Conditional Generation Grid - PMF-GMM", fontsize=28, fontweight='bold', y=1.00)
    
    gs_outer = gridspec.GridSpec(
        num_scales, len(TEMPERATURES) + 1, 
        figure=fig, 
        width_ratios=width_ratios,
        wspace=0.15,
        hspace=0.35,
        top=0.92,
    )
    
    model_module = importlib.import_module("hybrid_v3_1x1_double")
    GeneralFlow = getattr(model_module, 'GeneralFlow')
    
    for s_idx, scale in enumerate(SCALES):
        print(f"Processing Scale {scale}...")
        
        ax_scale = fig.add_subplot(gs_outer[s_idx, 0])
        ax_scale.axis('off')
        ax_scale.text(0.3, 0.5, f"Scale {scale}", fontsize=24, fontweight='bold', 
                      va='center', ha='center', rotation=90)
        
        model_path = f"../experiments/models/GMM/{VERSION}/best_loss_{scale}_hybrid_v3_1x1_double_CheckerboardGMM_Adam_0.5_0.1.pth"
        prior = CheckerboardGMM(total_dim=TOTAL_DIM, arr_num_classes=ARR_NUM_CLASSES, device=DEVICE, scale=scale, fixed_means=True)
        model = GeneralFlow(dropout_p=0.1).to(DEVICE)
        
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            prior.load_state_dict(checkpoint['prior_state_dict'])
            prior.means = checkpoint['means']
            model.eval()
            prior.eval()
        except FileNotFoundError:
            print(f"  -> Checkpoint not found for scale {scale}. Skipping...")
            continue
        
        with torch.no_grad():
            for temp_idx, temp in enumerate(TEMPERATURES):
                num_samples = SAMPLES_PER_TEMP[temp]
                
                gs_inner = gridspec.GridSpecFromSubplotSpec(
                    num_targets, num_samples, 
                    subplot_spec=gs_outer[s_idx, temp_idx + 1], 
                    wspace=0.05,
                    hspace=0.05
                )
                
                for t_idx, target in enumerate(TARGETS):
                    
                    z_parts = []
                    for k, c in enumerate(target):
                        mean_k = prior.means[k][c].to(DEVICE).unsqueeze(0).expand(num_samples, -1)
                        z_parts.append(mean_k)
                    
                    z = prior.get_full_latent(z_parts)
                    if temp > 0.0:
                        z = z + torch.randn_like(z) * temp
                        
                    z_structural = z.view(num_samples, 12, 14, 28)
                    img_gen = model.inverse(z_structural)
                    
                    for i in range(num_samples):
                        ax = fig.add_subplot(gs_inner[t_idx, i])
                        
                        img_np = img_gen[i].cpu().numpy() + 0.5 
                        img_np = np.transpose(img_np, (1, 2, 0)) 
                        img_np = np.clip(img_np, 0, 1)
                        
                        ax.imshow(img_np)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        
                        if s_idx == 0 and t_idx == 0 and i == (num_samples // 2):
                            ax.set_title(f"T = {temp}", fontsize=22, fontweight='bold', pad=20)

    save_path = f"{OUTPUT_DIR}/cond_gen_GMM_{VERSION}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Grid successfully generated and saved to: {save_path}")

if __name__ == "__main__":
    generate_and_plot_grid()
