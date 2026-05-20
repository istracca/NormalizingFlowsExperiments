import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from utils import set_seed

MNIST_NPZ_PATH = "SingleMNIST/data/mnist_data.npz"
DOUBLE_MNIST_NPZ_PATH = "DoubleMNIST/data/balanced_double_mnist.npz"
COLORED_DOUBLE_MNIST_NPZ_PATH = "EasyColoredDoubleMNIST/data/easy_colored_double_mnist.npz"

DATA_KEY = 'X_train' 

BASE_SAMPLES = 6                                        

def load_dataset_samples(path, shape, num_samples, is_rgb=False):
    """
    Attempts to load real data. Generates dummy data of the 
    correct shape if the path doesn't exist.
    """
    try:
        data = np.load(path)
        images = data[DATA_KEY]
        
        indices = np.random.choice(len(images), num_samples, replace=False)
        samples = images[indices]
        
        if samples.ndim == 2:
            if is_rgb:
                samples = samples.reshape(num_samples, shape[0], shape[1], 3)
            else:
                samples = samples.reshape(num_samples, shape[0], shape[1])
                
        elif not is_rgb and samples.ndim == 4 and samples.shape[-1] == 1:
            samples = samples.squeeze(axis=-1)
            
        return samples
        
    except FileNotFoundError:
        print(f"Warning: Could not find {path}. Generating dummy data.")
        if is_rgb:
            return np.random.rand(num_samples, shape[0], shape[1], 3)
        else:
            return np.random.rand(num_samples, shape[0], shape[1])

set_seed(42)
mnist_samples = load_dataset_samples(MNIST_NPZ_PATH, (28, 28), BASE_SAMPLES * 2, is_rgb=False)
double_samples = load_dataset_samples(DOUBLE_MNIST_NPZ_PATH, (28, 56), BASE_SAMPLES, is_rgb=False)
colored_samples = load_dataset_samples(COLORED_DOUBLE_MNIST_NPZ_PATH, (28, 56), BASE_SAMPLES, is_rgb=True)

fig = plt.figure(figsize=(10, 5.5))

gs = gridspec.GridSpec(nrows=3, ncols=BASE_SAMPLES * 2, figure=fig, hspace=0.4, wspace=0.1)

ax_row0 = fig.add_subplot(gs[0, :])
ax_row0.axis('off')
ax_row0.set_title("MNIST", fontsize=16, fontweight='bold', pad=1)

for i in range(BASE_SAMPLES * 2):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(mnist_samples[i], cmap='gray', vmin=0, vmax=1 if mnist_samples.dtype == np.float32 else 255)
    ax.axis('off')

ax_row1 = fig.add_subplot(gs[1, :])
ax_row1.axis('off')
ax_row1.set_title("DoubleMNIST", fontsize=16, fontweight='bold', pad=1)

for i in range(BASE_SAMPLES):
    ax = fig.add_subplot(gs[1, i*2 : i*2+2])
    ax.imshow(double_samples[i], cmap='gray', vmin=0, vmax=1 if double_samples.dtype == np.float32 else 255)
    ax.axis('off')

ax_row2 = fig.add_subplot(gs[2, :])
ax_row2.axis('off')
ax_row2.set_title("ColoredDoubleMNIST", fontsize=16, fontweight='bold', pad=1)

for i in range(BASE_SAMPLES):
    ax = fig.add_subplot(gs[2, i*2 : i*2+2])
    ax.imshow(colored_samples[i], vmin=0, vmax=1 if colored_samples.dtype == np.float32 else 255)
    ax.axis('off')

plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)

output_filename = "dataset_samples.png"                           
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Figure successfully saved to {output_filename}")