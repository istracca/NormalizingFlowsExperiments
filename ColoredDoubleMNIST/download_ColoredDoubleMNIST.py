import sys
sys.path.append('..')
from utils import set_seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

set_seed(42)

# 1. Download and Split Source Pools
print("Downloading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist['data'], mnist['target'].astype(np.int64)

# Normalize to 0-1 for easier color multiplication
X = X / 255.0

# Maintain official benchmark separation
X_train_val_pool, y_train_val_pool = X[:60000], y[:60000]
X_test_pool, y_test_pool = X[60000:], y[60000:]

# Create internal validation pool
X_train_pool, X_val_pool, y_train_pool, y_val_pool = train_test_split(
    X_train_val_pool, y_train_val_pool, test_size=10000, random_state=42, stratify=y_train_val_pool
)

def apply_random_color(batch_imgs):
    """
    Input: batch_imgs of shape (N, 28, 28) with values 0-1
    Output: colored_imgs of shape (N, 28, 28, 3)
    """
    N, H, W = batch_imgs.shape
    
    # Generate random RGB colors for each image in the batch: Shape (N, 1, 1, 3)
    # Uniform distribution [0, 1] ensures total randomness (uncorrelated)
    random_colors = np.random.rand(N, 1, 1, 3)
    
    # Reshape input images to broadcast: (N, 28, 28, 1)
    imgs_expanded = batch_imgs[:, :, :, np.newaxis]
    
    # Multiply: Black background (0) stays 0. Digit intensity scales the color.
    colored_imgs = imgs_expanded * random_colors
    
    return colored_imgs

def generate_colored_balanced_double_mnist(X_src, y_src, total_samples):
    X_imgs = X_src.reshape(-1, 28, 28)
    samples_per_comb = total_samples // 100
    
    # Pre-group indices by digit
    digit_indices = {d: np.where(y_src == d)[0] for d in range(10)}
    
    X_double_list = []
    y_double_list = []
    
    print(f"Generating {total_samples} samples...")
    
    for left_digit in range(10):
        for right_digit in range(10):
            # 1. Select Random Instances
            idx_left = np.random.choice(digit_indices[left_digit], samples_per_comb, replace=True)
            idx_right = np.random.choice(digit_indices[right_digit], samples_per_comb, replace=True)
            
            raw_left = X_imgs[idx_left]   # (Batch, 28, 28)
            raw_right = X_imgs[idx_right] # (Batch, 28, 28)
            
            # 2. Apply Random Colors INDEPENDENTLY
            # This ensures Left Color is not correlated with Right Color
            colored_left = apply_random_color(raw_left)   # (Batch, 28, 28, 3)
            colored_right = apply_random_color(raw_right) # (Batch, 28, 28, 3)
            
            # 3. Concatenate Horizontally
            # Result shape: (Batch, 28, 56, 3)
            combined_imgs = np.concatenate([colored_left, colored_right], axis=2)
            
            X_double_list.append(combined_imgs)
            
            labels = np.full((samples_per_comb, 2), [left_digit, right_digit])
            y_double_list.append(labels)
            
    X_final = np.vstack(X_double_list)
    y_final = np.vstack(y_double_list)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_final))
    
    # Return 4D array (N, 28, 56, 3) instead of flattened
    return X_final[shuffle_idx], y_final[shuffle_idx]

# 2. Generate Balanced Datasets
X_train, y_train = generate_colored_balanced_double_mnist(X_train_pool, y_train_pool, 50000)
X_val, y_val     = generate_colored_balanced_double_mnist(X_val_pool, y_val_pool, 10000)
X_test, y_test   = generate_colored_balanced_double_mnist(X_test_pool, y_test_pool, 10000)

# 3. Save as Float32 (0.0 to 1.0)
print("Saving to file...")
np.savez_compressed('data/colored_double_mnist.npz', 
                    X_train=X_train.astype(np.float32), y_train=y_train,
                    X_val=X_val.astype(np.float32), y_val=y_val,
                    X_test=X_test.astype(np.float32), y_test=y_test)

print(f"Train Shape: {X_train.shape}") # Should be (50000, 28, 56, 3)