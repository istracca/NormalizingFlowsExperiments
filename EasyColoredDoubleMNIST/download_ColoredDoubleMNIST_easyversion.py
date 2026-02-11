import sys
sys.path.append('..')
from utils import set_seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from scipy.special import softmax

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

BASE_COLORS = np.array([
    [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]
], dtype=np.float32)

def apply_random_color(batch_imgs):
    N, H, W = batch_imgs.shape
    
    # Pick base colors
    color_indices = np.random.choice(len(BASE_COLORS), size=N)
    chosen_colors = BASE_COLORS[color_indices]

    # Add noise
    noise = np.random.normal(loc=0.0, scale=0.1, size=(N, 3))
    noisy_colors_flat = np.clip(chosen_colors + noise, 0.0, 1.0)

    # Calculate Confidence: distance to all 7 base colors
    # We use -distance * 10.0 to create a sharp softmax probability
    dists = np.linalg.norm(noisy_colors_flat[:, np.newaxis, :] - BASE_COLORS, axis=2)
    confidences_all = softmax(-dists * 10.0, axis=1)
    # Get the confidence of the specific color we assigned
    confidences = confidences_all[np.arange(N), color_indices]

    # Final Image Construction
    noisy_colors_expanded = noisy_colors_flat[:, np.newaxis, np.newaxis, :]
    imgs_expanded = batch_imgs[:, :, :, np.newaxis]
    colored_imgs = imgs_expanded * noisy_colors_expanded
    
    return colored_imgs, color_indices, confidences

def generate_dataset(X_src, y_src, total_samples):
    X_imgs = X_src.reshape(-1, 28, 28)
    samples_per_comb = total_samples // 100
    digit_indices = {d: np.where(y_src == d)[0] for d in range(10)}
    
    X_list, y_digit_list, y_color_list = [], [], []
    all_confidences = []

    for left_digit in range(10):
        for right_digit in range(10):
            idx_l = np.random.choice(digit_indices[left_digit], samples_per_comb, replace=True)
            idx_r = np.random.choice(digit_indices[right_digit], samples_per_comb, replace=True)
            
            c_left, col_l, conf_l = apply_random_color(X_imgs[idx_l])
            c_right, col_r, conf_r = apply_random_color(X_imgs[idx_r])
            
            X_list.append(np.concatenate([c_left, c_right], axis=2))
            y_digit_list.append(np.full((samples_per_comb, 2), [left_digit, right_digit]))
            y_color_list.append(np.stack([col_l, col_r], axis=1))
            all_confidences.extend(conf_l)
            all_confidences.extend(conf_r)

    # Combine and Shuffle
    X_final = np.vstack(X_list)
    y_digits = np.vstack(y_digit_list)
    y_colors = np.vstack(y_color_list)
    
    idx = np.random.permutation(len(X_final))
    return X_final[idx], y_digits[idx], y_colors[idx], np.array(all_confidences)

# 2. Execution
X_train, y_train_d, y_train_c, train_conf = generate_dataset(X_train_pool, y_train_pool, 50000)
X_val, y_val_d, y_val_c, _ = generate_dataset(X_val_pool, y_val_pool, 10000)
X_test, y_test_d, y_test_c, _ = generate_dataset(X_test_pool, y_test_pool, 10000)

# 3. Save Plots
plt.figure(figsize=(8, 5))
plt.hist(train_conf, bins=50, color='royalblue', edgecolor='white')
plt.title("Color Assignment Confidence Distribution")
plt.xlabel("Confidence (Softmax Prob)")
plt.ylabel("Frequency")
plt.savefig('color_distribution.png')
print("Saved plot to color_distribution.png")

X_train, y_train_d, y_train_c, train_conf = generate_dataset(X_train_pool, y_train_pool, 50000)
X_val, y_val_d, y_val_c, _ = generate_dataset(X_val_pool, y_val_pool, 10000)
X_test, y_test_d, y_test_c, _ = generate_dataset(X_test_pool, y_test_pool, 10000)

print("Saving Version 1: Standard (Digits only)...")
np.savez_compressed('data/easy_colored_double_mnist.npz', 
                    X_train=X_train.astype(np.float32), y_train=y_train_d,
                    X_val=X_val.astype(np.float32), y_val=y_val_d,
                    X_test=X_test.astype(np.float32), y_test=y_test_d)

# --- FIX START ---
print("Saving Version 2: Enhanced (Digits + Color Attributes)...")

# Concatenate arrays to create shape (N, 4): [left_digit, right_digit, left_color, right_color]
y_train_combined = np.hstack([y_train_d, y_train_c])
y_val_combined = np.hstack([y_val_d, y_val_c])
y_test_combined = np.hstack([y_test_d, y_test_c])

# Save using the standard keys (X_train, y_train, etc.) just like Version 1
np.savez_compressed('data/easy_colored_double_mnist_with_attributes.npz', 
                    X_train=X_train.astype(np.float32), y_train=y_train_combined,
                    X_val=X_val.astype(np.float32), y_val=y_val_combined,
                    X_test=X_test.astype(np.float32), y_test=y_test_combined)
# --- FIX END ---

print("Both datasets have been saved successfully.")