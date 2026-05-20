import os
import sys
import torch
import torch.nn.functional as F
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse

sys.path.append(os.path.abspath(os.path.join('..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from utils import set_seed

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='4_attr', choices=['2_attr', '4_attr'], 
                    help='Dataset version to evaluate')
args = parser.parse_args()

VERSION = args.version
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
COND_DIM = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = f"samples_plots/Conditional/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_and_plot_grid():
    set_seed(42)

    num_targets = len(TARGETS)
    width_ratios = [0.25] + [SAMPLES_PER_TEMP[t] for t in TEMPERATURES]
    total_cols_visual = sum(width_ratios)
    
    fig = plt.figure(figsize=(total_cols_visual * 2.0, num_targets * 1 + 1))
    fig.suptitle(f"Conditional Generation Grid - CNF", fontsize=28, fontweight='bold', y=1.00)
    
    gs_outer = gridspec.GridSpec(
        1, len(TEMPERATURES) + 1, 
        figure=fig, 
        width_ratios=width_ratios,
        wspace=0.15, 
        hspace=0.35, 
        top=0.7,
    )
    
    print("Loading Conditional Generator...")
    model_path = f"../experiments/models/Conditional/{VERSION}/best_loss_conditional_scale_GaussianPrior_Adam_0.5_0.1.pth"
    model_module = importlib.import_module("conditional_scale")
    GeneralFlow = getattr(model_module, 'GeneralFlow')
    
    gen = GeneralFlow(num_classes=sum(ARR_NUM_CLASSES), dropout_p=0.1, cond_dim=COND_DIM).to(DEVICE)
    
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        gen.load_state_dict(checkpoint['model_state_dict'])
        gen.eval()
    except FileNotFoundError:
        print(f"Checkpoint not found at {model_path}. Exiting.")
        return
    
    print(f"Processing standard conditional model generation...")
    
    ax_approach = fig.add_subplot(gs_outer[0, 0])
    ax_approach.axis('off')
    ax_approach.text(0.3, 0.5, "Conditional", fontsize=24, fontweight='bold', 
                     va='center', ha='center', rotation=90)
    
    with torch.no_grad():
        for temp_idx, temp in enumerate(TEMPERATURES):
            num_samples = SAMPLES_PER_TEMP[temp]
            
            gs_inner = gridspec.GridSpecFromSubplotSpec(
                num_targets, num_samples, 
                subplot_spec=gs_outer[0, temp_idx + 1], 
                wspace=0.05, 
                hspace=0.05  
            )
            
            for t_idx, target in enumerate(TARGETS):
                
                y_onehot_list = []
                for k, c in enumerate(target):
                    lbls = torch.full((num_samples,), c, dtype=torch.long, device=DEVICE)
                    y_onehot = F.one_hot(lbls, num_classes=ARR_NUM_CLASSES[k]).float()
                    y_onehot_list.append(y_onehot)
                
                y_onehot_cond = torch.cat(y_onehot_list, dim=1)
                
                if temp > 0.0:
                    z = torch.randn(num_samples, 12, 14, 28, device=DEVICE) * temp
                else:
                    z = torch.zeros(num_samples, 12, 14, 28, device=DEVICE)
                    
                img_gen = gen.inverse(z, y_onehot_cond)
                
                for i in range(num_samples):
                    ax = fig.add_subplot(gs_inner[t_idx, i])
                    
                    img_np = img_gen[i].cpu().numpy() + 0.5 
                    img_np = np.transpose(img_np, (1, 2, 0)) 
                    img_np = np.clip(img_np, 0, 1)
                    
                    ax.imshow(img_np)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    if t_idx == 0 and i == (num_samples // 2):
                        ax.set_title(f"T = {temp}", fontsize=22, fontweight='bold', pad=20)

    save_path = f"{OUTPUT_DIR}/cond_gen_conditional_{VERSION}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Grid successfully generated and saved to: {save_path}")

if __name__ == "__main__":
    generate_and_plot_grid()