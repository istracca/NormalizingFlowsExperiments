from utils import set_seed
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

set_seed(42)

# 1. Download MNIST dataset (70,000 samples)
# mnist_784 is ordered: first 60k are training, last 10k are testing
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist['data'], mnist['target'].astype(np.int64)

# 2. Extract official Test Set (last 10,000)
X_test, y_test = X[60000:], y[60000:]

# 3. Use the first 60,000 for Training and Validation
X_train_full, y_train_full = X[:60000], y[:60000:]

# Split the 60k training set into Train (50k) and Val (10k)
# Stratify ensures the digit distribution remains consistent
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=10000, random_state=42, stratify=y_train_full
)

print(f"Training set:   {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set:       {X_test.shape[0]} samples (Official Benchmark)")

# 4. Save datasets as files
np.savez_compressed('mnist_data.npz',
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_test=X_test, y_test=y_test)

print("\nDatasets saved and partitioned correctly.")