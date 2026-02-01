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
from save_samples import save_samples_double
sys.path.append('../priors')
from CheckerboardGMM import CheckerboardGMM
from SimpleSplitGMM import SimpleSplitGMM
sys.path.append('../models')


set_seed(42)
parser = argparse.ArgumentParser(description='Train a flow-based model on MNIST.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale parameter for the prior')
parser.add_argument('--model', type=str, default='glow', help='Model name')
parser.add_argument('--prior', type=str, default='CheckerboardGMM', choices=['SimpleSplitGMM', 'CheckerboardGMM'], help='Prior type')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='Optimizer to use')
parser.add_argument('--transform', type=float, default=0.0, help='Percentage of data transformation to apply')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability for the model')
args = parser.parse_args()

SCALE = args.scale
MODEL = args.model
PRIOR = args.prior
OPTIMIZER = args.optimizer
TRANSFORM = args.transform
DROPOUT = args.dropout

module = importlib.import_module(MODEL)
GeneralFlow = getattr(module, 'GeneralFlow')

# Recover datasets from files
data = np.load('../data/balanced_double_mnist.npz')
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']
print("Datasets loaded from double_mnist_data.npz")
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

set_seed(42)
X_train_tensor = torch.tensor(X_train.reshape(-1, 1, 28, 56), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val.reshape(-1, 1, 28, 56), dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GeneralFlow(dropout_p=DROPOUT).to(device)
if PRIOR == 'CheckerboardGMM':
    prior = CheckerboardGMM(total_dim=1568, num_classes=10, num_attr=2, device=device, scale=SCALE, fixed_means=False).to(device)
elif PRIOR == 'SimpleSplitGMM':
    prior = SimpleSplitGMM(total_dim=1568, num_classes=10, num_attr=2, device=device, scale=SCALE, fixed_means=False).to(device)

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


os.makedirs(os.path.dirname(f'../experiments/logs/double/GMM/GMM_{SCALE}_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.log'), exist_ok=True)
# Set up logging to file
logging.basicConfig(
    filename=f'../experiments/logs/double/GMM/GMM_{SCALE}_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.log',
    filemode='w',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

os.makedirs(os.path.dirname(f'../experiments/csv/double/GMM/GMM_{SCALE}_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.csv'), exist_ok=True)
csv_path = f'../experiments/csv/double/GMM/GMM_{SCALE}_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.csv'
# clear existing file or create new one
if os.path.exists(csv_path):
    os.remove(csv_path)

headers = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr']
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
save_dir = f'../experiments/samples/double/GMM/GMM_{SCALE}_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}'
os.makedirs(save_dir, exist_ok=True)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, threshold=threshold_scheduler, threshold_mode='abs', verbose=True)
reduction_count = 0
previous_lr = optimizer.param_groups[0]['lr']
best_val_loss = float('inf')
best_acc = 0.0
epochs_with_enormous_loss = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
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
            batch_X = batch_X.view(-1, 1, 28, 56)

        z, sldj = model(batch_X)
        loss = prior.get_loss(z, sldj, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 5)

        optimizer.step()
        train_loss += loss.item()

        z_flat = z.view(z.size(0), -1)
        preds, _ = prior.classify(z_flat)
        if isinstance(preds, list):
            preds = torch.stack(preds, dim=1)
        elif isinstance(preds, tuple): # Handling your previous logic
            preds = torch.stack(preds[0], dim=1) if isinstance(preds[0], list) else preds[0]
        matched_rows = (preds == batch_y).all(dim=1)
        train_correct += matched_rows.sum().item()
        train_total += batch_y.size(0)
    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # dequantization
            batch_X = (batch_X * 255. + torch.rand_like(batch_X)) / 256.
            batch_X = batch_X - 0.5

            z, sldj = model(batch_X)
            loss = prior.get_loss(z, sldj, batch_y)
            val_loss += loss.item()

            z_flat = z.view(z.size(0), -1)
            preds, _ = prior.classify(z_flat)
            if isinstance(preds, list):
                preds = torch.stack(preds, dim=1)
            elif isinstance(preds, tuple): # Handling your previous logic
                preds = torch.stack(preds[0], dim=1) if isinstance(preds[0], list) else preds[0]
            matched_rows = (preds == batch_y).all(dim=1)
            val_correct += matched_rows.sum().item()
            val_total += batch_y.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

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

    os.makedirs(os.path.dirname(f'../experiments/models/double/GMM/best_loss_{SCALE}_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.pth'), exist_ok=True)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'prior_state_dict': prior.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'means': prior.means,
            'epoch': epoch + 1
        }, f'../experiments/models/double/GMM/best_loss_{SCALE}_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.pth')

    if val_loss > threshold_val_loss:
        epochs_with_enormous_loss += 1
        if epochs_with_enormous_loss >= patience_val_loss:
            logger.info(f"Validation loss has been enormous for {patience_val_loss} consecutive epochs. Stopping training.")
            break
    else:
        epochs_with_enormous_loss = 0

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'prior_state_dict': prior.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'means': prior.means,
            'epoch': epoch + 1
        }, f'../experiments/models/double/GMM/best_acc_{SCALE}_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.pth')

    logger.info(
        f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
        f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}'
    )

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'lr': current_lr
        })

    if epoch % 10 == 0:
        save_samples_double(model, prior, device, epoch, save_dir=save_dir, temp=0)

    

torch.save({
    'model_state_dict': model.state_dict(),
    'prior_state_dict': prior.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'means': prior.means
}, f'../experiments/models/double/GMM/final_{SCALE}_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.pth')