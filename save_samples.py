import torch
import os
import matplotlib.pyplot as plt

def save_samples(model, prior, device, epoch, save_dir, temp=0.0):

    os.makedirs(save_dir, exist_ok=True)

    print("Generazione Immagini...")
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
    print(f"Generazione Immagini Double (Epoch {epoch})...")
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