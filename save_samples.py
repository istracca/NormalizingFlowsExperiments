import torch
import os
import matplotlib.pyplot as plt
import numpy as np

def save_samples(model, prior, device, epoch, save_dir, temp=0.0):

    os.makedirs(save_dir, exist_ok=True)

    print("Generating images...")
    model.eval()

    device = next(model.parameters()).device 

    targets = list(range(10))

    with torch.no_grad():
        for target in targets:
            z = prior.means[target].unsqueeze(0).to(device) + torch.randn(1, prior.means.shape[1]).to(device) * temp

            z_structural = z.view(1, 4, 14, 14)
            img_gen = model.inverse(z_structural)
            
            plt.imshow(img_gen.squeeze().cpu(), cmap='gray')
            plt.axis('off')
            plt.savefig(
                f"{save_dir}/{target}.png", 
                bbox_inches='tight', 
                pad_inches=0
            )
            plt.close()

def save_samples_double(model, prior, device, epoch, save_dir, temp=0.0):
    """
    Generates a grid of samples by crossing all categories of attribute 0 
    with all categories of attribute 1.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Generating Double Images (Epoch {epoch})...")
    model.eval()
    
    # We assume prior.num_attr = 2
    # means[0] is for attribute 1, means[1] is for attribute 2
    num_cat_attr0 = prior.means[0].shape[0]  # e.g., 10
    num_cat_attr1 = prior.means[1].shape[0]  # e.g., 10
    
    with torch.no_grad():
        # Create a large plot to hold the grid
        fig, axes = plt.subplots(num_cat_attr0, num_cat_attr1, figsize=(num_cat_attr1, num_cat_attr0))
        
        for i in range(num_cat_attr0):
            for j in range(num_cat_attr1):
                # Combine the mean of cat 'i' from attr 0 and cat 'j' from attr 1
                mean_0 = prior.means[0][i]
                mean_1 = prior.means[1][j]
                
                z = prior.get_full_latent([mean_0.unsqueeze(0), mean_1.unsqueeze(0)])
                
                # Add noise (temperature)
                if temp > 0:
                    z = z + torch.randn_like(z) * temp
                
                # Reshape based on your model's expected structural input
                # Note: ensure (4 * 14 * 14) matches your total z dimensions
                z_structural = z.view(1, 4, 14, 28)
                img_gen = model.inverse(z_structural)
                
                # Plot in the grid
                ax = axes[i, j]
                ax.imshow(img_gen.squeeze().cpu().numpy(), cmap='gray')
                ax.axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(
            f"{save_dir}/epoch_{epoch}_grid.png", 
            bbox_inches='tight', 
            pad_inches=0.1
        )
        plt.close()


def save_samples_double_colored(model, prior, device, num_attr, epoch, save_dir, temp=0.0):
    """
    Generates sample grids.
    - If num_attr == 2: Generates one 10x10 grid (Digits).
    - If num_attr == 4: Generates TWO grids:
        1. Digit Grid (10x10) with fixed colors.
        2. Color Grid (7x7) with fixed digits.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # --- Helper to generate and save a single grid ---
    def generate_grid(row_attr_idx, col_attr_idx, fixed_defaults, filename_suffix):
        """
        row_attr_idx: Index of attribute varying along rows (Y-axis)
        col_attr_idx: Index of attribute varying along cols (X-axis)
        fixed_defaults: List of indices for all attributes. 
                        We will overwrite the row/col indices in the loop.
        """
        print(f"Generating {filename_suffix} (Epoch {epoch})...")
        
        # Get number of categories for the chosen axes
        num_rows = prior.means[row_attr_idx].shape[0]
        num_cols = prior.means[col_attr_idx].shape[0]
        
        with torch.no_grad():
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
            
            for i in range(num_rows):
                for j in range(num_cols):
                    
                    # 1. Build the Latent Vector Parts
                    z_parts = []
                    
                    for k in range(num_attr):
                        if k == row_attr_idx:
                            # Use the loop variable 'i' for the row attribute
                            part = prior.means[k][i].unsqueeze(0)
                        elif k == col_attr_idx:
                            # Use the loop variable 'j' for the col attribute
                            part = prior.means[k][j].unsqueeze(0)
                        else:
                            # Use the fixed default for other attributes
                            # fixed_defaults[k] should be an integer index (e.g., 0)
                            idx = fixed_defaults[k]
                            part = prior.means[k][idx].unsqueeze(0)
                        
                        z_parts.append(part)
                    
                    # 2. Combine and Sample
                    z = prior.get_full_latent(z_parts)
                    
                    if temp > 0:
                        z = z + torch.randn_like(z) * temp
                    
                    # 3. Reshape and Inverse
                    # Colored MNIST: 3 channels. 28*56*3 / (14*28) = 12 channels
                    z_structural = z.view(1, 12, 14, 28)
                    img_gen = model.inverse(z_structural)
                    
                    # 4. Post-processing (Shift + Permute + Clip)
                    img_gen = img_gen + 0.5 
                    img_np = img_gen.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    img_np = np.clip(img_np, 0, 1)
                    
                    ax = axes[i, j]
                    ax.imshow(img_np)
                    ax.axis('off')

            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(
                f"{save_dir}/epoch_{epoch}_{filename_suffix}.png", 
                bbox_inches='tight', 
                pad_inches=0.1
            )
            plt.close()

    # --- PLOT 1: Digits Grid (Vary Attr 0 and 1) ---
    # We fix colors (Attr 2, 3) to index 0 (Red)
    # Default indices for [DigitL, DigitR, ColorL, ColorR] -> [Loop, Loop, 0, 0]
    defaults_digits = [0] * num_attr 
    generate_grid(
        row_attr_idx=0, 
        col_attr_idx=1, 
        fixed_defaults=defaults_digits, 
        filename_suffix="digits_fixed_color"
    )

    # --- PLOT 2: Colors Grid (Only if we have 4 attributes) ---
    if num_attr >= 4:
        # Vary Attr 2 (Color L) and 3 (Color R)
        # If dataset is small, ensure index exists. 0 is always safe.
        defaults_colors = [0] * num_attr
        
        generate_grid(
            row_attr_idx=2, 
            col_attr_idx=3, 
            fixed_defaults=defaults_colors, 
            filename_suffix="colors_fixed_digit"
        )