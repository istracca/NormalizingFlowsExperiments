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
import kornia.augmentation as K
import sys
sys.path.append('../..')
from utils import set_seed
sys.path.append('../models')

set_seed(42)
parser = argparse.ArgumentParser(description='Train a discriminative classifier on MNIST.')
parser.add_argument('--model', type=str, default='disc_v3_double', help='Model name')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='Optimizer to use')
parser.add_argument('--transform', type=float, default=0.0, help='Percentage of data transformation to apply')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability for the model')
args = parser.parse_args()

MODEL = args.model
OPTIMIZER = args.optimizer
TRANSFORM = args.transform
DROPOUT = args.dropout

module = importlib.import_module(MODEL)
PseudoResNet = getattr(module, 'PseudoResNet')

# Recover datasets from files
data = np.load('../data/balanced_double_mnist.npz')
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']
print("Datasets loaded from balanced_double_mnist.npz")
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
model = PseudoResNet(num_classes = 10, dropout_p=DROPOUT).to(device)
criterion = torch.nn.CrossEntropyLoss()

# gpu_transform = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)).to(device)
gpu_transform = K.RandomAffine(degrees=10, translate=(0.1, 0.1), p=1.0).to(device)

if OPTIMIZER == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
elif OPTIMIZER == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-5)

num_epochs = 30000
max_reductions = 10
patience = 10
factor = 0.5
patience_val_loss = 10
threshold_val_loss = 1e5
threshold_scheduler = 1e-5

model_dir = '../experiments/models/Disc'
log_dir = '../experiments/logs/Disc'
csv_dir = '../experiments/csv/Disc'

for d in [model_dir, log_dir, csv_dir]:
    os.makedirs(d, exist_ok=True)

# Set up logging to file
logging.basicConfig(
    filename=f'{log_dir}/Disc_{MODEL}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.log',
    filemode='w',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()


headers = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr']
csv_path = f'{csv_dir}/Disc_{MODEL}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.csv'
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

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

        logits1, logits2 = model(batch_X)
        loss = criterion(logits1, batch_y[:, 0]) + criterion(logits2, batch_y[:, 1])
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        train_loss += loss.item()

        pred1 = logits1.argmax(dim=1)
        pred2 = logits2.argmax(dim=1)
        correct = ((pred1 == batch_y[:, 0]) & (pred2 == batch_y[:, 1])).sum().item()

        train_correct += correct
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

            logits1, logits2 = model(batch_X)
            loss = criterion(logits1, batch_y[:, 0]) + criterion(logits2, batch_y[:, 1])
            val_loss += loss.item()

            pred1 = logits1.argmax(dim=1)
            pred2 = logits2.argmax(dim=1)
            correct = ((pred1 == batch_y[:, 0]) & (pred2 == batch_y[:, 1])).sum().item()
            val_correct += correct
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

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1
        }, f'{model_dir}/best_loss_{MODEL}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.pth')
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
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1
        }, f'{model_dir}/best_acc_{MODEL}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.pth')

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


torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, f'{model_dir}/final_{MODEL}_{OPTIMIZER}_{TRANSFORM}.pth')