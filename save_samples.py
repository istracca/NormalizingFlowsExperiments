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

def save_samples_conditional_colored(model, prior, device, arr_num_classes, epoch, save_dir, temp=0.0):
    
    os.makedirs(save_dir, exist_ok=True)
    num_attr = len(arr_num_classes)

    print("Generating conditional images...")
    model.eval()

    def generate_grid(row_attr_idx, col_attr_idx, fixed_defaults, filename_suffix):
        """
        row_attr_idx: Index of attribute varying along rows (Y-axis)
        col_attr_idx: Index of attribute varying along cols (X-axis)
        fixed_defaults: List of indices for all attributes. 
                        We will overwrite the row/col indices in the loop.
        """
        print(f"Generating {filename_suffix} (Epoch {epoch})...")
        
        num_rows = arr_num_classes[row_attr_idx]
        num_cols = arr_num_classes[col_attr_idx]
        
        with torch.no_grad():
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
            
            for i in range(num_rows):
                for j in range(num_cols):
                    
                    z = torch.zeros(1, prior.total_dim).to(device) + torch.randn(1, prior.total_dim).to(device) * temp
                    one_hots = []

                    for k in range(num_attr):
                        if k == row_attr_idx:
                            c_onehot = torch.nn.functional.one_hot(torch.tensor([i]), num_classes=arr_num_classes[k]).float().to(device)
                        elif k == col_attr_idx:
                            c_onehot = torch.nn.functional.one_hot(torch.tensor([j]), num_classes=arr_num_classes[k]).float().to(device)
                        else:
                            c_onehot = torch.nn.functional.one_hot(torch.tensor([fixed_defaults[k]]), num_classes=arr_num_classes[k]).float().to(device)
                        
                        one_hots.append(c_onehot)
                    
                    c_onehot = torch.cat(one_hots, dim=1)
                    z_structural = z.view(1, 12, 14, 28)

                    img_gen = model.inverse(z_structural, c_onehot)
                    
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


    if num_attr == 1:
        defaults_digits = [0] * num_attr 
        generate_grid(
            row_attr_idx=0, 
            col_attr_idx=0, 
            fixed_defaults=defaults_digits, 
            filename_suffix="digits_fixed_color"
        )

    elif num_attr >= 2:
        defaults_digits = [0] * num_attr 
        generate_grid(
            row_attr_idx=0, 
            col_attr_idx=1, 
            fixed_defaults=defaults_digits, 
            filename_suffix="digits_fixed_color"
        )

    if num_attr ==3:
        defaults_colors = [0] * num_attr 
        generate_grid(
            row_attr_idx=2, 
            col_attr_idx=2, 
            fixed_defaults=defaults_colors, 
            filename_suffix="colors_fixed_digit"
        )

    elif num_attr == 4:
        defaults_colors = [0] * num_attr
        
        generate_grid(
            row_attr_idx=2, 
            col_attr_idx=3, 
            fixed_defaults=defaults_colors, 
            filename_suffix="colors_fixed_digit"
        )
            
def save_samples_conditional(model, prior, device, epoch, save_dir, temp=0.0):
    
    os.makedirs(save_dir, exist_ok=True)

    print("Generating conditional images...")
    model.eval()

    device = next(model.parameters()).device 

    targets = list(range(10))

    with torch.no_grad():
        for target in targets:
            z = torch.zeros(1, prior.total_dim).to(device) + torch.randn(1, prior.total_dim).to(device) * temp
            y_onehot = torch.nn.functional.one_hot(torch.tensor([target]), num_classes=prior.num_classes).float().to(device)

            z_structural = z.view(1, 4, 14, 14)
            img_gen = model.inverse(z_structural, y_onehot)
            
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
    
    num_cat_attr0 = prior.means[0].shape[0]            
    num_cat_attr1 = prior.means[1].shape[0]            
    
    with torch.no_grad():
        fig, axes = plt.subplots(num_cat_attr0, num_cat_attr1, figsize=(num_cat_attr1, num_cat_attr0))
        
        for i in range(num_cat_attr0):
            for j in range(num_cat_attr1):
                mean_0 = prior.means[0][i]
                mean_1 = prior.means[1][j]
                
                z = prior.get_full_latent([mean_0.unsqueeze(0), mean_1.unsqueeze(0)])
                
                if temp > 0:
                    z = z + torch.randn_like(z) * temp
                
                z_structural = z.view(1, 4, 14, 28)
                img_gen = model.inverse(z_structural)
                
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

def save_samples_double_gaussian(model, prior, device, epoch, save_dir, mode='independent', temp=0.0):
    """
    mode: 'independent' or 'combinatorial'
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Generating Images (Epoch {epoch}, Mode: {mode})...")
    model.eval()
    
    num_cat_attr0 = prior.independent_means[0].shape[0]  
    num_cat_attr1 = prior.independent_means[1].shape[0]  
    
    with torch.no_grad():
        fig, axes = plt.subplots(num_cat_attr0, num_cat_attr1, figsize=(num_cat_attr1, num_cat_attr0))
        
        for i in range(num_cat_attr0):
            for j in range(num_cat_attr1):
                
                if mode == 'independent':
                    mean_0 = prior.independent_means[0][i]
                    mean_1 = prior.independent_means[1][j]
                    z = prior.get_full_latent([mean_0.unsqueeze(0), mean_1.unsqueeze(0)])
                
                elif mode == 'combinatorial':
                    flat_idx = i * num_cat_attr1 + j
                    z = prior.combinatorial_means[flat_idx].unsqueeze(0)
                
                else:
                    raise ValueError("Mode must be 'independent' or 'combinatorial'")

                if temp > 0:
                    z = z + torch.randn_like(z) * temp
                
                z_structural = z.view(1, 4, 14, 28)
                img_gen = model.inverse(z_structural)
                
                axes[i, j].imshow(img_gen.squeeze().cpu().numpy(), cmap='gray')
                axes[i, j].axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f"{save_dir}/epoch_{epoch}_grid_{mode}.png", bbox_inches='tight', pad_inches=0.1)
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
    
    def generate_grid(row_attr_idx, col_attr_idx, fixed_defaults, filename_suffix):
        """
        row_attr_idx: Index of attribute varying along rows (Y-axis)
        col_attr_idx: Index of attribute varying along cols (X-axis)
        fixed_defaults: List of indices for all attributes. 
                        We will overwrite the row/col indices in the loop.
        """
        print(f"Generating {filename_suffix} (Epoch {epoch})...")
        
        num_rows = prior.means[row_attr_idx].shape[0]
        num_cols = prior.means[col_attr_idx].shape[0]
        
        with torch.no_grad():
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
            
            for i in range(num_rows):
                for j in range(num_cols):
                    
                    z_parts = []
                    
                    for k in range(num_attr):
                        if k == row_attr_idx:
                            part = prior.means[k][i].unsqueeze(0)
                        elif k == col_attr_idx:
                            part = prior.means[k][j].unsqueeze(0)
                        else:
                            idx = fixed_defaults[k]
                            part = prior.means[k][idx].unsqueeze(0)
                        
                        z_parts.append(part)
                    
                    z = prior.get_full_latent(z_parts)
                    
                    if temp > 0:
                        z = z + torch.randn_like(z) * temp
                    
                    z_structural = z.view(1, 12, 14, 28)
                    img_gen = model.inverse(z_structural)
                    
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


    if num_attr == 1:
        defaults_digits = [0] * num_attr 
        generate_grid(
            row_attr_idx=0, 
            col_attr_idx=0, 
            fixed_defaults=defaults_digits, 
            filename_suffix="digits_fixed_color"
        )

    elif num_attr >= 2:
        defaults_digits = [0] * num_attr 
        generate_grid(
            row_attr_idx=0, 
            col_attr_idx=1, 
            fixed_defaults=defaults_digits, 
            filename_suffix="digits_fixed_color"
        )

    if num_attr ==3:
        defaults_colors = [0] * num_attr 
        generate_grid(
            row_attr_idx=2, 
            col_attr_idx=2, 
            fixed_defaults=defaults_colors, 
            filename_suffix="colors_fixed_digit"
        )

    elif num_attr == 4:
        defaults_colors = [0] * num_attr
        
        generate_grid(
            row_attr_idx=2, 
            col_attr_idx=3, 
            fixed_defaults=defaults_colors, 
            filename_suffix="colors_fixed_digit"
        )


def save_samples_double_conditional(model, prior, device, epoch, save_dir, arr_num_classes, temp=0.0):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Generating Images (Epoch {epoch}...")
    model.eval()
    
    num_cat_attr0 = arr_num_classes[0]
    num_cat_attr1 = arr_num_classes[1]

    with torch.no_grad():
        fig, axes = plt.subplots(num_cat_attr0, num_cat_attr1, figsize=(num_cat_attr1, num_cat_attr0))
        
        for i in range(num_cat_attr0):
            for j in range(num_cat_attr1):
                z = torch.zeros(1, prior.total_dim).to(device) + torch.randn(1, prior.total_dim).to(device) * temp
                y_onehot = torch.cat([
                    torch.nn.functional.one_hot(torch.tensor([i]), num_classes=arr_num_classes[0]),
                    torch.nn.functional.one_hot(torch.tensor([j]), num_classes=arr_num_classes[1])
                ], dim=1).float().to(device)

                z_structural = z.view(1, 4, 14, 28)
                img_gen = model.inverse(z_structural, y_onehot)
                
                axes[i, j].imshow(img_gen.squeeze().cpu().numpy(), cmap='gray')
                axes[i, j].axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f"{save_dir}/epoch_{epoch}_grid.png", bbox_inches='tight', pad_inches=0.1)
        plt.close()


def save_samples_gaussian_colored(model, prior, device, arr_num_classes, epoch, save_dir, mode='independent', temp=0.0):
    """
    mode: 'independent' or 'combinatorial'
    """
    os.makedirs(save_dir, exist_ok=True)
    num_attr = len(arr_num_classes)

    print(f"Generating gaussian images (Epoch {epoch}, Mode: {mode})...")
    model.eval()

    def generate_grid(row_attr_idx, col_attr_idx, fixed_defaults, filename_suffix):
        """
        row_attr_idx: Index of attribute varying along rows (Y-axis)
        col_attr_idx: Index of attribute varying along cols (X-axis)
        fixed_defaults: List of indices for all attributes. 
        """
        print(f"Generating {filename_suffix} (Epoch {epoch})...")
        
        num_rows = arr_num_classes[row_attr_idx]
        num_cols = arr_num_classes[col_attr_idx]
        
        with torch.no_grad():
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
            
            if not isinstance(axes, np.ndarray):
                axes = np.array([[axes]])
            elif axes.ndim == 1:
                if num_rows == 1:
                    axes = axes[np.newaxis, :]
                else:
                    axes = axes[:, np.newaxis]

            for i in range(num_rows):
                for j in range(num_cols):
                    
                    current_indices = []
                    for k in range(num_attr):
                        if k == row_attr_idx:
                            current_indices.append(i)
                        elif k == col_attr_idx:
                            current_indices.append(j)
                        else:
                            current_indices.append(fixed_defaults[k])
                    
                    if mode == 'independent':
                        means = []
                        for k in range(num_attr):
                            means.append(prior.independent_means[k][current_indices[k]].unsqueeze(0))
                        z = prior.get_full_latent(means)
                        
                    elif mode == 'combinatorial':
                        flat_idx = 0
                        multiplier = 1
                        for k in reversed(range(num_attr)):
                            flat_idx += current_indices[k] * multiplier
                            multiplier *= arr_num_classes[k]
                        
                        z = prior.combinatorial_means[flat_idx].unsqueeze(0)
                        
                    else:
                        raise ValueError("Mode must be 'independent' or 'combinatorial'")

                    if temp > 0:
                        z = z + torch.randn_like(z) * temp
                    
                    z_structural = z.view(1, 12, 14, 28)
                    img_gen = model.inverse(z_structural)                                  
                    
                    img_gen = img_gen + 0.5 
                    img_np = img_gen.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    img_np = np.clip(img_np, 0, 1)
                    
                    ax = axes[i, j]
                    ax.imshow(img_np)
                    ax.axis('off')

            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(
                f"{save_dir}/epoch_{epoch}_{filename_suffix}_{mode}.png", 
                bbox_inches='tight', 
                pad_inches=0.1
            )
            plt.close()

    if num_attr == 1:
        defaults_digits = [0] * num_attr 
        generate_grid(
            row_attr_idx=0, 
            col_attr_idx=0, 
            fixed_defaults=defaults_digits, 
            filename_suffix="digits_fixed_color"
        )

    elif num_attr >= 2:
        defaults_digits = [0] * num_attr 
        generate_grid(
            row_attr_idx=0, 
            col_attr_idx=1, 
            fixed_defaults=defaults_digits, 
            filename_suffix="digits_fixed_color"
        )

    if num_attr == 3:
        defaults_colors = [0] * num_attr 
        generate_grid(
            row_attr_idx=2, 
            col_attr_idx=2, 
            fixed_defaults=defaults_colors, 
            filename_suffix="colors_fixed_digit"
        )

    elif num_attr == 4:
        defaults_colors = [0] * num_attr
        generate_grid(
            row_attr_idx=2, 
            col_attr_idx=3, 
            fixed_defaults=defaults_colors, 
            filename_suffix="colors_fixed_digit"
        )