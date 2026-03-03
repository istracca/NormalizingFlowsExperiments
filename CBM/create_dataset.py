import numpy as np
import os

def compute_task_label(concepts):
    """
    Computes y based on the rule:
    y = 0 if:
       (Sum(digits) is ODD AND Colors are DIFFERENT)
       OR
       (Sum(digits) is EVEN AND Colors are EQUAL)
    y = 1 otherwise.
    
    concepts shape: (N, 4) -> [left_digit, right_digit, left_color, right_color]
    """
    left_digit = concepts[:, 0]
    right_digit = concepts[:, 1]
    left_color = concepts[:, 2]
    right_color = concepts[:, 3]
    
    # Condition A: Sum is Odd
    digit_sum = left_digit + right_digit
    is_odd = (digit_sum % 2 != 0)
    
    # Condition B: Colors are Equal
    colors_equal = (left_color == right_color)
    
    # Define cases for y=0
    case_1 = (is_odd) & (~colors_equal)
    case_2 = (~is_odd) & (colors_equal)

    # Initialize y to 1 and set to 0 for the defined cases
    y = np.ones(len(concepts), dtype=np.int64)
    y[case_1 | case_2] = 0
    
    return y


input_path = '../EasyColoredDoubleMNIST/data/easy_colored_double_mnist_with_attributes.npz'
output_dir = 'data'
output_path = os.path.join(output_dir, 'double_mnist_cbm.npz')

if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input file not found at: {input_path}")

os.makedirs(output_dir, exist_ok=True)

print(f"Loading data from {input_path}...")
data = np.load(input_path)

X_train = data['X_train']
c_train = data['y_train'].astype(np.int64)

X_val = data['X_val']
c_val = data['y_val'].astype(np.int64)

X_test = data['X_test']
c_test = data['y_test'].astype(np.int64)

print(f"Data Loaded. Train shape: {X_train.shape}")
print(f"Attributes (c) shape: {c_train.shape}")

print("Computing task labels 'y' from attributes 'c'...")
y_train = compute_task_label(c_train)
y_val = compute_task_label(c_val)
y_test = compute_task_label(c_test)

print(f"Train Class Balance (y=0): {np.mean(y_train == 0):.2%}")
print(f"Train Class Balance (y=1): {np.mean(y_train == 1):.2%}")

print(f"Saving new CBM dataset to {output_path}...")
np.savez_compressed(
    output_path, 
    X_train=X_train, 
    c_train=c_train,
    y_train=y_train,
    X_val=X_val, 
    c_val=c_val,
    y_val=y_val,
    X_test=X_test, 
    c_test=c_test,
    y_test=y_test
)
print("Done.")