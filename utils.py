import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    """
    Set the random seed for reproducibility across various libraries.
    """
    # 1. Python standard
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. NumPy
    np.random.seed(seed)
    
    # 3. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # 4. Determinism in convolution algorithms (CUDNN)
    # Slightly slows down training but ensures that convolution operations 
    # are identical every time.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Seed set to: {seed}")


from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TransformedTensorDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, transform=None):
        self.x = x_tensor
        self.y = y_tensor
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y
        
    def __len__(self):
        return len(self.x)