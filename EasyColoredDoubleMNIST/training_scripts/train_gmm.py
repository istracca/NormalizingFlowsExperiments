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
from save_samples import save_samples_double_colored
sys.path.append('../priors')
from SimpleSplitGMM import SimpleSplitGMM
from CheckerboardGMM import CheckerboardGMM
sys.path.append('../models')


set_seed(42)
parser = argparse.ArgumentParser(description='Train a flow-based model on MNIST.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale parameter for the prior')
parser.add_argument('--model', type=str, default='hybrid_v3_1x1_double', help='Model name')
parser.add_argument('--prior', type=str, default='CheckerboardGMM', choices=['SimpleSplitGMM', 'CheckerboardGMM'], help='Prior type')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='Optimizer to use')
parser.add_argument('--transform', type=float, default=0.0, help='Percentage of data transformation to apply')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability for the model')
parser.add_argument('--version', type=str, default='2_attr', choices=['1_attr','2_attr', '3_attr', '4_attr'], help='Use 2 attributes (only digits) or 4 attributes (digits + colors)')
args = parser.parse_args()

SCALE = args.scale
MODEL = args.model
PRIOR = args.prior
OPTIMIZER = args.optimizer
TRANSFORM = args.transform
DROPOUT = args.dropout
VERSION = args.version
module = importlib.import_module(MODEL)
GeneralFlow = getattr(module, 'GeneralFlow')

if VERSION == '1_attr':
    data = np.load('../data/easy_colored_double_mnist.npz')
    arr_num_classes = [10]
elif VERSION == '2_attr':
    data = np.load('../data/easy_colored_double_mnist.npz')
    arr_num_classes = [10, 10]
elif VERSION == '3_attr':
    data = np.load('../data/easy_colored_double_mnist_with_attributes.npz')
    arr_num_classes = [10, 10, 7]
elif VERSION == '4_attr':
    data = np.load('../data/easy_colored_double_mnist_with_attributes.npz')
    arr_num_classes = [10, 10, 7, 7]
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

if VERSION == "1_attr":
    y_train = y_train[:, 0:1]
    y_val = y_val[:, 0:1]
    y_test = y_test[:, 0:1]
elif VERSION == "3_attr":
    y_train = y_train[:, 0:3]
    y_val = y_val[:, 0:3]
    y_test = y_test[:, 0:3]

print(f"Datasets loaded")
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

set_seed(42)
X_train_tensor = torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val.transpose(0, 3, 1, 2), dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GeneralFlow(dropout_p=DROPOUT).to(device)
if PRIOR == 'CheckerboardGMM':
    prior = CheckerboardGMM(total_dim=4704, arr_num_classes=arr_num_classes, device=device, scale=SCALE, fixed_means=True).to(device)
    save_samples_fun = save_samples_double_colored
elif PRIOR == 'SimpleSplitGMM':
    prior = SimpleSplitGMM(total_dim=4704, arr_num_classes=arr_num_classes, device=device, scale=SCALE, fixed_means=True).to(device)
    save_samples_fun = save_samples_double_colored

gpu_transform = K.RandomAffine(degrees=10, translate=(0.1, 0.1), p=1.0).to(device)

if OPTIMIZER == 'Adam':
    optimizer = optim.Adam(list(model.parameters()) + list(prior.parameters()), lr=1e-4)
elif OPTIMIZER == 'SGD':
    optimizer = optim.SGD(list(model.parameters()) + list(prior.parameters()), lr=1e-4, momentum=0.9, weight_decay=1e-5)

print(list(prior.parameters()))
print(list(prior.means))

num_epochs = 30000
max_reductions = 10
patience = 10
factor = 0.5
patience_val_loss = 10
threshold_val_loss = 1e5
threshold_scheduler = 0.5


os.makedirs(os.path.dirname(f'../experiments/logs/GMM/{VERSION}/GMM_{SCALE}_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.log'), exist_ok=True)
logging.basicConfig(
    filename=f'../experiments/logs/GMM/{VERSION}/GMM_{SCALE}_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.log',
    filemode='w',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

os.makedirs(os.path.dirname(f'../experiments/csv/GMM/{VERSION}/GMM_{SCALE}_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.csv'), exist_ok=True)
csv_path = f'../experiments/csv/GMM/{VERSION}/GMM_{SCALE}_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.csv'
if os.path.exists(csv_path):
    os.remove(csv_path)

num_attr = len(arr_num_classes)
headers = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr']
headers = headers + [f'attr_{i}_train_acc' for i in range(num_attr)] + [f'attr_{i}_val_acc' for i in range(num_attr)]
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
save_dir = f'../experiments/samples/GMM/{VERSION}/GMM_{SCALE}_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}'
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
    train_correct_per_attr = 0 
    train_total = 0
    
    for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        if TRANSFORM > 0:
            n = batch_X.size(0)
            n_transform = int(TRANSFORM * n)
            if n_transform > 0:
                idx = torch.randperm(n, device=batch_X.device)[:n_transform]
                batch_X[idx] = gpu_transform(batch_X[idx])
                
        batch_X = (batch_X * 255. + torch.rand_like(batch_X)) / 256.
        batch_X = batch_X - 0.5

        optimizer.zero_grad()
        if batch_X.dim() == 2:
            batch_X = batch_X.view(-1, 3, 28, 56)

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
        elif isinstance(preds, tuple): 
            preds = torch.stack(preds[0], dim=1) if isinstance(preds[0], list) else preds[0]
            
        matched_rows = (preds == batch_y).all(dim=1)
        train_correct += matched_rows.sum().item()
        
        if isinstance(train_correct_per_attr, int):
            train_correct_per_attr = torch.zeros(batch_y.size(1), device=device)
        train_correct_per_attr += (preds == batch_y).sum(dim=0)
        
        train_total += batch_y.size(0)
        
    train_loss /= len(train_loader)
    train_acc = train_correct / train_total
    train_acc_per_attr = (train_correct_per_attr / train_total).tolist()

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_correct_per_attr = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            batch_X = (batch_X * 255. + torch.rand_like(batch_X)) / 256.
            batch_X = batch_X - 0.5

            z, sldj = model(batch_X)
            loss = prior.get_loss(z, sldj, batch_y)
            val_loss += loss.item()

            z_flat = z.view(z.size(0), -1)
            preds, _ = prior.classify(z_flat)
            if isinstance(preds, list):
                preds = torch.stack(preds, dim=1)
            elif isinstance(preds, tuple): 
                preds = torch.stack(preds[0], dim=1) if isinstance(preds[0], list) else preds[0]
                
            matched_rows = (preds == batch_y).all(dim=1)
            val_correct += matched_rows.sum().item()
            
            if isinstance(val_correct_per_attr, int):
                val_correct_per_attr = torch.zeros(batch_y.size(1), device=device)
            val_correct_per_attr += (preds == batch_y).sum(dim=0)
            
            val_total += batch_y.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    val_acc_per_attr = (val_correct_per_attr / val_total).tolist()

    scheduler.step(train_loss)

    current_lr = optimizer.param_groups[0]['lr']
    if current_lr < previous_lr:
        reduction_count += 1
        previous_lr = current_lr
        logger.info(f"Reduction {reduction_count}/{max_reductions}: LR dropped to {current_lr}")

    if reduction_count >= max_reductions:
        logger.info(f"Breaking loop: Learning rate reduced more than {max_reductions} times.")
        break

    os.makedirs(os.path.dirname(f'../experiments/models/GMM/{VERSION}/best_loss_{SCALE}_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.pth'), exist_ok=True)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'prior_state_dict': prior.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'means': prior.means,
            'epoch': epoch + 1
        }, f'../experiments/models/GMM/{VERSION}/best_loss_{SCALE}_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.pth')

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
        }, f'../experiments/models/GMM/{VERSION}/best_acc_{SCALE}_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.pth')

    logger.info(
        f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
        f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}\n'
        f'Train Acc per Attr: {[round(x, 4) for x in train_acc_per_attr]}\n'
        f'Val Acc per Attr:   {[round(x, 4) for x in val_acc_per_attr]}'
    )

    row_data = {
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'lr': current_lr
    }
    for i, acc in enumerate(train_acc_per_attr):
        row_data[f'attr_{i}_train_acc'] = acc
    for i, acc in enumerate(val_acc_per_attr):
        row_data[f'attr_{i}_val_acc'] = acc

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow(row_data)

    if epoch % 10 == 0:
        save_samples_fun(model, prior, device, num_attr=len(arr_num_classes), epoch=epoch, save_dir=save_dir, temp=0)

torch.save({
    'model_state_dict': model.state_dict(),
    'prior_state_dict': prior.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'means': prior.means
}, f'../experiments/models/GMM/{VERSION}/final_{SCALE}_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}.pth')