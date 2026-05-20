import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
import argparse
import importlib
import csv
import os
from torchvision import transforms
import kornia.augmentation as K
import sys
sys.path.append('../..')
from utils import set_seed
from save_samples import save_samples_conditional
sys.path.append('../priors')
from GaussianPrior import GaussianPrior
sys.path.append('../models')

set_seed(42)
parser = argparse.ArgumentParser(description='Train a flow-based model on MNIST.')
parser.add_argument('--model', type=str, default='conditional_hybrid_v3_1x1', help='Model name')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='Optimizer to use')
parser.add_argument('--prior', type=str, default='GaussianPrior', help='Prior to use')
parser.add_argument('--transform', type=float, default=0.5, help='Percentage of data transformation to apply')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability for the model')
parser.add_argument('--cond_dim', type=int, default=64, help='Embedding dimension for the conditional information')
args = parser.parse_args()

MODEL = args.model
OPTIMIZER = args.optimizer
PRIOR = args.prior
TRANSFORM = args.transform
DROPOUT = args.dropout
COND_DIM = args.cond_dim
num_classes = 10

os.makedirs('../experiments/models/Conditional', exist_ok=True)
os.makedirs('../experiments/logs/Conditional', exist_ok=True)
os.makedirs('../experiments/csv/Conditional', exist_ok=True)
save_dir = f'../experiments/samples/Conditional/Conditional_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}'
os.makedirs(save_dir, exist_ok=True)

module = importlib.import_module(MODEL)
GeneralFlow = getattr(module, 'GeneralFlow')

# Recover datasets from files
data = np.load('../data/mnist_data.npz')
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']
print("Datasets loaded from mnist_data.npz")

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

set_seed(42)
X_train_tensor = torch.tensor(X_train.reshape(-1, 1, 28, 28), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val.reshape(-1, 1, 28, 28), dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GeneralFlow(dropout_p=DROPOUT, num_classes=num_classes, cond_dim=COND_DIM).to(device)
prior = GaussianPrior(device=device, total_dim=784, num_classes=num_classes)

# gpu_transform = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)).to(device)
gpu_transform = K.RandomAffine(degrees=10, translate=(0.1, 0.1), p=1.0).to(device)

if OPTIMIZER == 'Adam':
    optimizer = optim.Adam(list(model.parameters()) + list(prior.parameters()), lr=1e-4)
elif OPTIMIZER == 'SGD':
    optimizer = optim.SGD(list(model.parameters()) + list(prior.parameters()), lr=1e-4, momentum=0.9, weight_decay=1e-5)

print(list(prior.parameters()))

num_epochs = 30000
max_reductions = 10
patience = 10
factor = 0.5
patience_val_loss = 10
threshold_val_loss = 1e5
threshold_scheduler = 0.5

# Set up logging to file
logging.basicConfig(
    filename=f'../experiments/logs/Conditional/Conditional_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.log',
    filemode='w',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

csv_path = f'../experiments/csv/Conditional/Conditional_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.csv'
# clear existing file or create new one
if os.path.exists(csv_path):
    os.remove(csv_path)

headers = ['epoch', 'train_loss', 'val_loss', 'lr']
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, threshold=threshold_scheduler, threshold_mode='abs', verbose=True)
reduction_count = 0
previous_lr = optimizer.param_groups[0]['lr']
best_val_loss = float('inf')
epochs_with_enormous_loss = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        if TRANSFORM > 0:
            n = batch_X.size(0)
            n_transform = int(TRANSFORM * n)
            if n_transform > 0:
                idx = torch.randperm(n, device=batch_X.device)[:n_transform]
                batch_X[idx] = gpu_transform(batch_X[idx])
        
        # dequantization
        batch_X = (batch_X * 255. + torch.rand_like(batch_X)) / 256.
        batch_X = batch_X - 0.5

        optimizer.zero_grad()
        if batch_X.dim() == 2:
            batch_X = batch_X.view(-1, 1, 28, 28)

        # transform batch_y in one-hot encoding
        batch_y_onehot = torch.nn.functional.one_hot(batch_y, num_classes=num_classes).float()

        z, sldj = model(batch_X, batch_y_onehot)

        loss = prior.get_loss(z, sldj, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 5)

        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # dequantization
            batch_X = (batch_X * 255. + torch.rand_like(batch_X)) / 256.
            batch_X = batch_X - 0.5


            batch_y_onehot = torch.nn.functional.one_hot(batch_y, num_classes=num_classes).float()
            z, sldj = model(batch_X, batch_y_onehot)
            loss = prior.get_loss(z, sldj, batch_y)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    scheduler.step(train_loss)

    # Check if the learning rate was reduced
    current_lr = optimizer.param_groups[0]['lr']
    if current_lr < previous_lr:
        reduction_count += 1
        previous_lr = current_lr
        logger.info(f"Reduction {reduction_count}/{max_reductions}: LR dropped to {current_lr}")

    # Break the loop if threshold is met
    if reduction_count >= max_reductions:
        logger.info(f"Breaking loop: Learning rate reduced more than {max_reductions} times.")
        break

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'prior_state_dict': prior.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'means': prior.means,
            'epoch': epoch + 1
        }, f'../experiments/models/Conditional/best_loss_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.pth')

    if val_loss > threshold_val_loss:
        epochs_with_enormous_loss += 1
        if epochs_with_enormous_loss >= patience_val_loss:
            logger.info(f"Validation loss has been enormous for {patience_val_loss} consecutive epochs. Stopping training.")
            break
    else:
        epochs_with_enormous_loss = 0

    logger.info(
        f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
    )

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr
        })

    if epoch % 10 == 0:
        save_samples_conditional(model, prior, device, epoch, save_dir=save_dir + f'/epoch_{epoch}', temp=0)

    

torch.save({
    'model_state_dict': model.state_dict(),
    'prior_state_dict': prior.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'means': prior.means
}, f'../experiments/models/Conditional/final_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.pth')
