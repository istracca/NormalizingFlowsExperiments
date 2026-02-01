from utils import set_seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

set_seed(42)

# 1. Download and Split Source Pools
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist['data'], mnist['target'].astype(np.int64)

# Maintain official benchmark separation
X_train_val_pool, y_train_val_pool = X[:60000], y[:60000:]
X_test_pool, y_test_pool = X[60000:], y[60000:]

# Create internal validation pool
X_train_pool, X_val_pool, y_train_pool, y_val_pool = train_test_split(
    X_train_val_pool, y_train_val_pool, test_size=10000, random_state=42, stratify=y_train_val_pool
)

def generate_balanced_double_mnist(X_src, y_src, total_samples):
    X_imgs = X_src.reshape(-1, 28, 28)
    samples_per_comb = total_samples // 100
    
    # Pre-group indices by digit
    digit_indices = {d: np.where(y_src == d)[0] for d in range(10)}
    
    X_double_list = []
    y_double_list = []
    
    for left_digit in range(10):
        for right_digit in range(10):
            # RANDOM selection of specific handwriting instances from the digit pool
            idx_left = np.random.choice(digit_indices[left_digit], samples_per_comb, replace=True)
            idx_right = np.random.choice(digit_indices[right_digit], samples_per_comb, replace=True)
            
            combined_imgs = np.concatenate([X_imgs[idx_left], X_imgs[idx_right]], axis=2)
            X_double_list.append(combined_imgs)
            
            labels = np.full((samples_per_comb, 2), [left_digit, right_digit])
            y_double_list.append(labels)
            
    X_final = np.vstack(X_double_list)
    y_final = np.vstack(y_double_list)
    
    # Shuffle so the model doesn't see batches of the same combination
    shuffle_idx = np.random.permutation(len(X_final))
    return X_final[shuffle_idx].reshape(len(X_final), -1), y_final[shuffle_idx]

# 2. Generate Balanced Datasets
X_train, y_train = generate_balanced_double_mnist(X_train_pool, y_train_pool, 50000)
X_val, y_val     = generate_balanced_double_mnist(X_val_pool, y_val_pool, 10000)
X_test, y_test   = generate_balanced_double_mnist(X_test_pool, y_test_pool, 10000)

# 3. Save
np.savez_compressed('balanced_double_mnist.npz', 
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_test=X_test, y_test=y_test)

print(f"Double MNIST Training set:   {X_train.shape} (Labels: {y_train.shape})")
print(f"Double MNIST Validation set: {X_val.shape}")
print(f"Double MNIST Test set:       {X_test.shape}")

def verify_balance(y_labels, name="Dataset"):
    # Convert [left, right] pairs to 0-99 integers for easy counting
    pair_ids = y_labels[:, 0] * 10 + y_labels[:, 1]
    unique, counts = np.unique(pair_ids, return_counts=True)
    
    print(f"\n--- {name} Balance Check ---")
    print(f"Total Unique Combinations: {len(unique)} (Expected: 100)")
    print(f"Minimum samples/combination: {counts.min()}")
    print(f"Maximum samples/combination: {counts.max()}")
    print(f"Mean samples/combination: {counts.mean()}")

verify_balance(y_train, "Training")
verify_balance(y_test, "Testing")