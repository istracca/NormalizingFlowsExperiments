import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# --- Symmetric 7-Color Palette (RGB Cube Corners) ---
COLOR_PALETTE = np.array([
    # -- Primaries (1.0, 0.0, 0.0) permutations --
    [1.0, 0.0, 0.0],  # 0: Red
    [0.0, 1.0, 0.0],  # 1: Green
    [0.0, 0.0, 1.0],  # 2: Blue
    
    # -- Secondaries (0.0, 1.0, 1.0) permutations --
    [1.0, 1.0, 0.0],  # 3: Yellow
    [0.0, 1.0, 1.0],  # 4: Cyan
    [1.0, 0.0, 1.0],  # 5: Magenta
    
    # -- Achromatic (All channels max) --
    [1.0, 1.0, 1.0],  # 6: White
])

COLOR_NAMES = [
    'Red', 'Green', 'Blue', 
    'Yellow', 'Cyan', 'Magenta', 
    'White'
]

def get_dominant_color_index(img_crop, threshold=0.1):
    """
    Extracts the mean color of the non-black pixels in an image crop
    and finds the nearest neighbor in the COLOR_PALETTE.
    """
    # img_crop shape: (28, 28, 3)
    
    # 1. Mask out the black background
    # We consider a pixel "lit" if its max channel value > threshold
    mask = np.max(img_crop, axis=-1) > threshold
    
    if mask.sum() == 0:
        # If the digit is empty/too faint (rare), default to White (9) or index 0
        return 9 
        
    # 2. Get the average RGB of the digit pixels
    # shape: (N_lit_pixels, 3) -> mean -> (3,)
    avg_color = img_crop[mask].mean(axis=0)
    
    # 3. Find nearest bucket (Euclidean distance)
    # expand_dims to make shapes compatible for cdist
    # cdist returns distance matrix, we take argmin
    dists = cdist([avg_color], COLOR_PALETTE, metric='euclidean')
    return np.argmin(dists)

def process_dataset(X, y):
    """
    X: (N, 28, 56, 3)
    y: (N, 2) -> [left_digit, right_digit]
    
    Returns:
    y_new: (N, 4) -> [left_digit, right_digit, left_color, right_color]
    """
    N = X.shape[0]
    y_colors = np.zeros((N, 2), dtype=np.int64)
    
    print(f"Processing {N} samples...")
    
    for i in range(N):
        img = X[i] # (28, 56, 3)
        
        # Split Left and Right
        left_crop = img[:, :28, :]
        right_crop = img[:, 28:, :]
        
        # Identify Color Class
        c_left = get_dominant_color_index(left_crop)
        c_right = get_dominant_color_index(right_crop)
        
        y_colors[i] = [c_left, c_right]
        
    # Concatenate original labels with new color labels
    y_new = np.hstack([y, y_colors])
    return y_new

# --- 2. Load Data ---
print("Loading colored_double_mnist.npz...")
data = np.load('data/colored_double_mnist.npz')

X_train = data['X_train']
y_train = data['y_train']
X_val   = data['X_val']
y_val   = data['y_val']
X_test  = data['X_test']
y_test  = data['y_test']

# --- 3. Process Labels ---
print("\n--- Processing Training Set ---")
y_train_new = process_dataset(X_train, y_train)

print("\n--- Processing Validation Set ---")
y_val_new = process_dataset(X_val, y_val)

print("\n--- Processing Test Set ---")
y_test_new = process_dataset(X_test, y_test)

# --- 5. Save New Dataset ---
save_path = 'data/colored_double_mnist_with_attributes.npz'
print(f"\nSaving new dataset to {save_path}...")

np.savez_compressed(
    save_path, 
    X_train=X_train, y_train=y_train_new,
    X_val=X_val, y_val=y_val_new,
    X_test=X_test, y_test=y_test_new
)

print("Done. New label shape is (N, 4): [L_digit, R_digit, L_color, R_color]")