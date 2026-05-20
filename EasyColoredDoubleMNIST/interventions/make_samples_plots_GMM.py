import os
import sys
import torch
import importlib
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

# Ensure utility scripts can be imported
sys.path.append(os.path.abspath(os.path.join('..', '..')))
sys.path.append(os.path.join(os.path.dirname(__file__), '../priors'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))
from utils import set_seed
from CheckerboardGMM import CheckerboardGMM

# ==========================================
# 1. CONFIGURATION 
# ==========================================
parser = argparse.ArgumentParser(description='Evaluate Progression of Interventions.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale parameter')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
args = parser.parse_args()

SCALE = args.scale

MODEL_CONFIG = {
    "MODEL": "hybrid_v3_1x1_double", 
    "PRIOR": "CheckerboardGMM",
    "OPTIMIZER": "Adam",
    "TRANSFORM": 0.5,
    "DROPOUT": 0.1
}

MODEL_CONFIG["PATH"] = (
    f"../experiments/models/GMM/2_attr/best_loss_{SCALE}_{MODEL_CONFIG['MODEL']}_{MODEL_CONFIG['PRIOR']}_"
    f"{MODEL_CONFIG['OPTIMIZER']}_{MODEL_CONFIG['TRANSFORM']}_{MODEL_CONFIG['DROPOUT']}.pth"
)

DISC_CONFIG = {
    "MODEL": "disc_v3_double",
    "PATH": f"../experiments/models/Disc/2_attr/best_loss_disc_v3_double_Adam_0.5_0.1.pth",
    "DROPOUT": 0.1
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOT_DIR = f"plots/GMM_Intervention_Grids_Scale_{SCALE}/"

LAMBDAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

os.makedirs(PLOT_DIR, exist_ok=True)
for f in os.listdir(PLOT_DIR):
    os.remove(os.path.join(PLOT_DIR, f))

BASE_COLORS = torch.tensor([
    [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]
], dtype=torch.float32)
COLOR_NAMES = ["Red", "Green", "Blue", "Yellow", "Cyan", "Magenta", "White"]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def parse_predictions(preds_raw):
    if isinstance(preds_raw, list):
        return torch.stack(preds_raw, dim=1)
    elif isinstance(preds_raw, tuple):
        return torch.stack(preds_raw[0], dim=1) if isinstance(preds_raw[0], list) else preds_raw[0]
    return preds_raw

def get_dominant_colors(imgs, device):
    imgs_shifted = imgs + 0.5 
    left_half = imgs_shifted[:, :, :, :28]
    right_half = imgs_shifted[:, :, :, 28:]
    
    mask_left = (left_half.max(dim=1, keepdim=True)[0] > 0.2).float()
    mask_right = (right_half.max(dim=1, keepdim=True)[0] > 0.2).float()
    
    mean_left = (left_half * mask_left).sum(dim=(2, 3)) / (mask_left.sum(dim=(2, 3)) + 1e-5)
    mean_right = (right_half * mask_right).sum(dim=(2, 3)) / (mask_right.sum(dim=(2, 3)) + 1e-5)
    
    base_colors = BASE_COLORS.to(device)
    color_left = torch.argmin(torch.cdist(mean_left, base_colors), dim=1)
    color_right = torch.argmin(torch.cdist(mean_right, base_colors), dim=1)
    
    return color_left, color_right

def plot_progression_grid(plot_data_dict, title, filename, plot_type):
    # Filter out empty rows to dynamically size the plot
    valid_rows = [data for r, data in sorted(plot_data_dict.items()) if data is not None]
    num_rows = len(valid_rows)
    
    if num_rows == 0:
        print(f"Skipping plot '{title}' - No matching samples found.")
        return

    cols = len(LAMBDAS) + 1  # 1 Original + lambdas
    fig, axes = plt.subplots(num_rows, cols, figsize=(4.5 * cols, 3.4 * num_rows))
    
    # Ensure axes is a 2D array even if there's only 1 row
    if num_rows == 1:
        axes = np.array([axes])
        
    fig.suptitle(title, fontsize=32, fontweight='bold', y=0.98)
    
    for row_idx, data in enumerate(valid_rows):
        orig_ax = axes[row_idx, 0]
        orig_rgb = np.clip(data['orig_img'].transpose(1, 2, 0), 0, 1)
        orig_ax.imshow(orig_rgb)
        
        orig_st_l = COLOR_NAMES[data['orig_style'][0]][:3]
        orig_st_r = COLOR_NAMES[data['orig_style'][1]][:3]
        orig_pr = data['orig_preds']
        tgts = data['targets']
        
        orig_ax.axis('off')
        
        # Replace set_title with manual text placement for Original column
        orig_ax.text(0.5, 1.28, f"Sty: [{orig_st_l}, {orig_st_r}]", ha='center', va='bottom', transform=orig_ax.transAxes, fontsize=28)
        orig_ax.text(0.5, 1.05, f"{orig_pr} -> {tgts}", ha='center', va='bottom', transform=orig_ax.transAxes, fontsize=28)
        
        orig_ax.plot([1.19, 1.19], [-0.2, 1.2], color='black', linestyle='--', 
            linewidth=4, transform=orig_ax.transAxes, clip_on=False)
        
        # Add big bold separated header for Original column
        if row_idx == 0:
            orig_ax.text(0.5, 1.5, "Original", transform=orig_ax.transAxes, 
                         ha='center', va='bottom', fontsize=40, fontweight='bold')

        for col_idx, lam in enumerate(LAMBDAS):
            ax = axes[row_idx, col_idx + 1]
            interv_rgb = np.clip(data['imgs'][col_idx].transpose(1, 2, 0) + 0.5, 0, 1)
            ax.imshow(interv_rgb)
            ax.axis('off')
            
            st_l = COLOR_NAMES[data['detected_styles'][col_idx][0]][:3]
            st_r = COLOR_NAMES[data['detected_styles'][col_idx][1]][:3]
            pr = data['preds'][col_idx]
            dp = data['disc_preds'][col_idx]
            
            sty_text = f"Sty:[{st_l}, {st_r}]"
            pmf_text = f"PMF:{pr}"
            disc_text = f"Disc:{dp}"
            
            # Check success condition based on plot type
            style_success = (data['detected_styles'][col_idx][0] == data['orig_style'][0]) and \
                            (data['detected_styles'][col_idx][1] == data['orig_style'][1])
            cc_success = (pr == tgts)
            disc_success = (dp == tgts)

            # Draw Custom Colored & Bolded Title Parts
            if plot_type == "style":
                color = 'green' if style_success else 'red'
                ax.text(0.5, 1.28 if num_rows > 1 else 1.23, sty_text, ha='center', va='bottom', transform=ax.transAxes, fontsize=28, color=color, fontweight='bold')
                ax.text(0.5, 1.05 if num_rows > 1 else 1.00, f"{pmf_text} | {disc_text}", ha='center', va='bottom', transform=ax.transAxes, fontsize=28, color='black')
                
            elif plot_type == "cc":
                color = 'green' if cc_success else 'red'
                ax.text(0.5, 1.28 if num_rows > 1 else 1.23, sty_text, ha='center', va='bottom', transform=ax.transAxes, fontsize=28, color='black')
                ax.text(0.48, 1.05 if num_rows > 1 else 1.00, pmf_text, ha='right', va='bottom', transform=ax.transAxes, fontsize=28, color=color, fontweight='bold')
                ax.text(0.50, 1.05 if num_rows > 1 else 1.00, " | ", ha='center', va='bottom', transform=ax.transAxes, fontsize=28, color='black')
                ax.text(0.52, 1.05 if num_rows > 1 else 1.00, disc_text, ha='left', va='bottom', transform=ax.transAxes, fontsize=28, color='black')
                
            elif plot_type == "disc":
                color = 'green' if disc_success else 'red'
                ax.text(0.5, 1.28 if num_rows > 1 else 1.23, sty_text, ha='center', va='bottom', transform=ax.transAxes, fontsize=28, color='black')
                ax.text(0.48, 1.05 if num_rows > 1 else 1.00, pmf_text, ha='right', va='bottom', transform=ax.transAxes, fontsize=28, color='black')
                ax.text(0.50, 1.05 if num_rows > 1 else 1.00, " | ", ha='center', va='bottom', transform=ax.transAxes, fontsize=28, color='black')
                ax.text(0.52, 1.05 if num_rows > 1 else 1.00, disc_text, ha='left', va='bottom', transform=ax.transAxes, fontsize=28, color=color, fontweight='bold')
            
            # Add big bold separated header for Lambda columns
            if row_idx == 0:
                ax.text(0.5, 1.5, f"$\\lambda={lam}$", transform=ax.transAxes, 
                        ha='center', va='bottom', fontsize=40, fontweight='bold')
                
    plt.tight_layout(rect=[0, 0, 1, 0.99] if num_rows > 1 else [0, 0, 1, 0.9], h_pad=0.2, w_pad=1.5) 
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
# ==========================================
# 3. SEARCH & EVALUATION LOOP
# ==========================================
def find_and_plot_progressions(model, prior, disc_model, loader, device):
    model.eval()
    prior.eval()
    disc_model.eval() 
    
    # Store candidates as lists to pick optimal ones later
    plots = {
        "1_interv": { "style": {r: [] for r in range(1, 7)}, "cc": {r: [] for r in range(1, 7)}, "disc": {r: [] for r in range(1, 7)} },
        "2_interv": { "style": {r: [] for r in range(1, 7)}, "cc": {r: [] for r in range(1, 7)}, "disc": {r: [] for r in range(1, 7)} }
    }
    
    def all_found():
        # Stop early only if we have at least 3 candidates per cell for selection
        for group in ["1_interv", "2_interv"]:
            for cat in ["style", "cc", "disc"]:
                for r in range(1, 7):
                    if len(plots[group][cat][r]) < 3:
                        return False
        return True

    print("Searching for interventions matching conditions...")
    with torch.no_grad():
        for batch_X, batch_y in tqdm(loader, desc="Batches"):
            if all_found(): break
            
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            B = batch_X.size(0)
            
            batch_X = (batch_X * 255.0 + torch.rand_like(batch_X)) / 256.0
            batch_X_centered = batch_X - 0.5
            
            orig_color_left, orig_color_right = get_dominant_colors(batch_X_centered, device)
            z, _ = model(batch_X_centered)
            z_flat = z.view(B, -1)
            
            preds_pre_raw, _ = prior.classify(z_flat)
            preds_pre = parse_predictions(preds_pre_raw)
            
            z_parts = prior.get_parts(z_flat)
            left_z, right_z = z_parts[0], z_parts[1]
            
            mu_left_orig = prior.means[0][preds_pre[:, 0]].to(device)
            mu_right_orig = prior.means[1][preds_pre[:, 1]].to(device)
            
            for offset0 in range(10):
                for offset1 in range(10):
                    if all_found(): break
                    if offset0 == 0 and offset1 == 0: continue
                    
                    t0 = (preds_pre[:, 0] + offset0) % 10
                    t1 = (preds_pre[:, 1] + offset1) % 10
                    
                    if offset0 != 0 and offset1 == 0: int_type = "attr0"
                    elif offset0 == 0 and offset1 != 0: int_type = "attr1"
                    else: int_type = "both"
                    
                    int_group = "1_interv" if int_type in ["attr0", "attr1"] else "2_interv"
                        
                    mu_left_target = prior.means[0][t0].to(device)
                    mu_right_target = prior.means[1][t1].to(device)

                    z_interv_list = []
                    for lam in LAMBDAS:
                        left_z_shifted = mu_left_target + lam * (left_z - mu_left_orig)
                        right_z_shifted = mu_right_target + lam * (right_z - mu_right_orig)
                        
                        if int_type == "attr0": z_interv = prior.get_full_latent([left_z_shifted, right_z])
                        elif int_type == "attr1": z_interv = prior.get_full_latent([left_z, right_z_shifted])
                        elif int_type == "both": z_interv = prior.get_full_latent([left_z_shifted, right_z_shifted])
                            
                        z_interv_list.append(z_interv)

                    z_interv_batched = torch.cat(z_interv_list, dim=0)
                    z_interv_structural = z_interv_batched.view(len(LAMBDAS) * B, 12, 14, 28)
                    img_interv_batched = model.inverse(z_interv_structural)

                    preds_post_raw, _ = prior.classify(z_interv_structural.view(len(LAMBDAS) * B, -1))
                    preds_post_batched = parse_predictions(preds_post_raw)
                    interv_color_left, interv_color_right = get_dominant_colors(img_interv_batched, device)
                    
                    disc_logits = disc_model(img_interv_batched)
                    disc_preds_batched = torch.stack([logit.argmax(dim=1) for logit in disc_logits], dim=1)
                    
                    L = len(LAMBDAS)
                    preds_post = preds_post_batched.view(L, B, 2).transpose(0, 1)      
                    disc_preds = disc_preds_batched.view(L, B, 2).transpose(0, 1)      
                    col_left = interv_color_left.view(L, B).transpose(0, 1)            
                    col_right = interv_color_right.view(L, B).transpose(0, 1)          
                    imgs = img_interv_batched.view(L, B, 3, 28, 56).transpose(0, 1)    

                    cc_match = (preds_post[:, :, 0] == t0.unsqueeze(1)) & (preds_post[:, :, 1] == t1.unsqueeze(1))
                    style_match = (col_left == orig_color_left.unsqueeze(1)) & (col_right == orig_color_right.unsqueeze(1))
                    disc_match = (disc_preds[:, :, 0] == t0.unsqueeze(1)) & (disc_preds[:, :, 1] == t1.unsqueeze(1))

                    def extract(idx):
                        return {
                            'orig_img': batch_X[idx].cpu().numpy(),
                            'orig_style': [orig_color_left[idx].item(), orig_color_right[idx].item()],
                            'orig_preds': preds_pre[idx].cpu().tolist(),
                            'targets': [t0[idx].item(), t1[idx].item()],
                            'imgs': imgs[idx].cpu().numpy(),
                            'detected_styles': [[col_left[idx, l].item(), col_right[idx, l].item()] for l in range(L)],
                            'preds': preds_post[idx].cpu().tolist(),
                            'disc_preds': disc_preds[idx].cpu().tolist()
                        }

                    for i in range(B):
                        cc = cc_match[i].cpu().numpy()
                        st = style_match[i].cpu().numpy()
                        dc = disc_match[i].cpu().numpy()
                        
                        has_white = 6 in (orig_color_left[i].item(), orig_color_right[i].item())

                        def try_add(cat, r):
                            # Force alternating rows for 1_intervention
                            if int_group == "1_interv":
                                if r % 2 != 0 and int_type != "attr0":
                                    return  # Reject attr1 for odd rows
                                if r % 2 == 0 and int_type != "attr1":
                                    return  # Reject attr0 for even rows

                            if len(plots[int_group][cat][r]) < 10:
                                plots[int_group][cat][r].append({
                                    'data': extract(i),
                                    'int_type': int_type,
                                    'has_white': has_white
                                })
                        # Style vs Lambda Decay
                        if all(cc) and all(st): try_add("style", 1)
                        elif not st[0] and all(st[1:]): try_add("style", 2)
                        elif not any(st[:2]) and all(st[2:]): try_add("style", 3)
                        elif not any(st[:3]) and all(st[3:]): try_add("style", 4)
                        elif not any(st[:4]) and all(st[4:]): try_add("style", 5)
                        elif not any(st[:5]) and st[5]: try_add("style", 6)

                        # CC vs Lambda Decay
                        if all(cc): try_add("cc", 1)
                        elif all(cc[:5]) and not cc[5]: try_add("cc", 2)
                        elif all(cc[:4]) and not any(cc[4:]): try_add("cc", 3)
                        elif all(cc[:3]) and not any(cc[3:]): try_add("cc", 4)
                        elif all(cc[:2]) and not any(cc[2:]): try_add("cc", 5)
                        elif cc[0] and not any(cc[1:]): try_add("cc", 6)

                        # Disc Accuracy vs Lambda Decay
                        if all(dc): try_add("disc", 1)
                        elif all(dc[:5]) and not dc[5]: try_add("disc", 2)
                        elif all(dc[:4]) and not any(dc[4:]): try_add("disc", 3)
                        elif all(dc[:3]) and not any(dc[3:]): try_add("disc", 4)
                        elif all(dc[:2]) and not any(dc[2:]): try_add("disc", 5)
                        elif dc[0] and not any(dc[1:]): try_add("disc", 6)

    print("Selecting best candidates to balance attributes and avoid white colors...")
    final_plots = { "1_interv": {}, "2_interv": {} }
    
    for grp in ["1_interv", "2_interv"]:
        for cat in ["style", "cc", "disc"]:
            final_plots[grp][cat] = {r: None for r in range(1, 7)}
            attr0_count = 0
            attr1_count = 0
            
            for r in range(1, 7):
                candidates = plots[grp][cat][r]
                if not candidates: continue
                
                # Scoring function to prioritize selection: lower score is better
                def score(c):
                    s = 0
                    if c['has_white']: 
                        s += 10  # Heavily penalize white colors
                    
                    if grp == "1_interv":
                        # Penalize the attribute type that has been picked more often to force a balance
                        if c['int_type'] == "attr0" and attr0_count > attr1_count: s += 5
                        if c['int_type'] == "attr1" and attr1_count > attr0_count: s += 5
                    return s
                
                candidates.sort(key=score)
                best_candidate = candidates[0]
                
                final_plots[grp][cat][r] = best_candidate['data']
                
                if best_candidate['int_type'] == "attr0": attr0_count += 1
                elif best_candidate['int_type'] == "attr1": attr1_count += 1

    print("Generating Final Plots...")
    for grp_name, grp_title in [("1_interv", "1 Intervention"), ("2_interv", "2 Interventions")]:
        plot_progression_grid(final_plots[grp_name]["style"], f"Style vs Lambda ({grp_title})", os.path.join(PLOT_DIR, f"plot_1_style_{grp_name}_GMM_{SCALE}.png"), "style")
        plot_progression_grid(final_plots[grp_name]["cc"], f"Cyclic Consistency vs Lambda ({grp_title})", os.path.join(PLOT_DIR, f"plot_2_cc_{grp_name}_GMM_{SCALE}.png"), "cc")
        plot_progression_grid(final_plots[grp_name]["disc"], f"Disc Accuracy vs Lambda ({grp_title})", os.path.join(PLOT_DIR, f"plot_3_disc_{grp_name}_GMM_{SCALE}.png"), "disc")
    print("Done!")

if __name__ == "__main__":
    set_seed(42)
    
    print("Loading Generative Classifier...")
    model_module = importlib.import_module(MODEL_CONFIG['MODEL'])
    GeneralFlow = getattr(model_module, 'GeneralFlow')
    
    # Prior instantiation
    prior_class = globals()[MODEL_CONFIG['PRIOR']]
    prior = prior_class(total_dim=4704, arr_num_classes=[10, 10], scale=SCALE, device=DEVICE, fixed_means=True)
    prior.num_classes = 20
    
    model = GeneralFlow(dropout_p=MODEL_CONFIG['DROPOUT']).to(DEVICE)
    checkpoint = torch.load(MODEL_CONFIG['PATH'], map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    prior.load_state_dict(checkpoint['prior_state_dict'])
    prior.means = checkpoint['means']
    prior.num_attr = 2
    prior.total_dim = 4704
    
    print("Loading Discriminative Classifier...")
    disc_module = importlib.import_module(DISC_CONFIG['MODEL'])
    PseudoResNet = getattr(disc_module, 'PseudoResNet')
    
    disc_model = PseudoResNet(arr_num_classes=[10, 10], in_channels=3, dropout_p=DISC_CONFIG['DROPOUT'], device=DEVICE).to(DEVICE)
    disc_checkpoint = torch.load(DISC_CONFIG['PATH'], map_location=DEVICE)
    disc_model.load_state_dict(disc_checkpoint['model_state_dict'])
    
    from torch.utils.data import TensorDataset, DataLoader
    
    print("Loading test dataset...")
    data = np.load('../data/easy_colored_double_mnist.npz')
    X_test, y_test = data['X_test'], data['y_test']
    y_test = y_test[:, 0:2]
    
    X_test_tensor = torch.tensor(X_test.transpose(0, 3, 1, 2), dtype=torch.float32)[:512]
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)[:512]
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    if len(test_loader) > 0:
        find_and_plot_progressions(model, prior, disc_model, test_loader, DEVICE)