import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import sys
import os
import importlib
import csv
import logging
import kornia.augmentation as K

# --- IMPORTS ---
sys.path.append('../..') 
sys.path.append('../models')
sys.path.append('../priors')
from utils import set_seed
from save_samples import save_samples_double_colored
from cbm import FlowGMMConceptExtractor, TaskPredictor, ModularCBM
from hybrid_v3_1x1_double import GeneralFlow 
from CheckerboardGMM import CheckerboardGMM
from SimpleSplitGMM import SimpleSplitGMM

parser = argparse.ArgumentParser(description='Train CBM Joint Paradigm')
parser.add_argument('--scale', type=float, default=1.0, help='Scale parameter for the prior')
parser.add_argument('--model', type=str, default='hybrid_v3_1x1_double', help='Model name')
parser.add_argument('--prior', type=str, default='CheckerboardGMM', choices=['SimpleSplitGMM', 'CheckerboardGMM'], help='Prior type')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='Optimizer to use')
parser.add_argument('--transform', type=float, default=0.0, help='Percentage of data transformation to apply')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability for the model')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lambda_c', type=float, default=1.0, help='Weight for concept loss')
args = parser.parse_args()

# Constants
SCALE = args.scale
MODEL = args.model
PRIOR = args.prior
OPTIMIZER = args.optimizer
TRANSFORM = args.transform
DROPOUT = args.dropout
LAMBDA_C = args.lambda_c

module = importlib.import_module(args.model)
GeneralFlow = getattr(module, 'GeneralFlow')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(42)


# ==========================================
# 1. SETUP LOGGING & DIRECTORIES
# ==========================================
experiment_name = f"CBM_joint_lam{LAMBDA_C}_{SCALE}_{MODEL}_{PRIOR}_{OPTIMIZER}_{TRANSFORM}_{DROPOUT}"
base_dir = '../experiments'

# 1. Logs
log_dir = f'{base_dir}/logs/CBM/joint/lambda_{LAMBDA_C}'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=f'{log_dir}/{experiment_name}.log',
    filemode='w',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# 2. CSV
csv_dir = f'{base_dir}/csv/CBM/joint/lambda_{LAMBDA_C}'
os.makedirs(csv_dir, exist_ok=True)
csv_path = f'{csv_dir}/{experiment_name}.csv'

if os.path.exists(csv_path):
    os.remove(csv_path)

headers = ['epoch', 
           'train_loss_total', 'train_loss_c', 'train_loss_y', 
           'val_loss_total', 'val_loss_c', 'val_loss_y',
           'train_acc_c', 'train_acc_y', 
           'val_acc_c', 'val_acc_y', 
           'lr']
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()

# 3. Models & Samples
model_save_dir = f'{base_dir}/models/CBM/joint/lambda_{LAMBDA_C}'
sample_save_dir = f'{base_dir}/samples/CBM/joint/lambda_{LAMBDA_C}/{experiment_name}'
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(sample_save_dir, exist_ok=True)

logger.info(f"Starting experiment: {experiment_name}")


# ==========================================
# 2. LOAD DATASET
# ==========================================
logger.info("Loading CBM Dataset...")
data_path = '../data/double_mnist_cbm.npz'
data = np.load(data_path)

X_train = torch.tensor(data['X_train'], dtype=torch.float32).permute(0, 3, 1, 2)
c_train = torch.tensor(data['c_train'], dtype=torch.long)
y_train = torch.tensor(data['y_train'], dtype=torch.long)

X_val = torch.tensor(data['X_val'], dtype=torch.float32).permute(0, 3, 1, 2)
c_val = torch.tensor(data['c_val'], dtype=torch.long)
y_val = torch.tensor(data['y_val'], dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, c_train, y_train), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, c_val, y_val), batch_size=128, shuffle=False)


# ==========================================
# 3. INITIALIZE MODELS (FROM SCRATCH)
# ==========================================
arr_num_classes = [10, 10, 7, 7] 

# 1. Flow
flow_model = GeneralFlow(dropout_p=args.dropout).to(device)

# 2. Prior
if args.prior == 'CheckerboardGMM':
    prior = CheckerboardGMM(
        total_dim=4704, arr_num_classes=arr_num_classes, num_attr=len(arr_num_classes), 
        device=device, scale=args.scale, fixed_means=False
    ).to(device)
elif args.prior == 'SimpleSplitGMM':
    prior = SimpleSplitGMM(
        total_dim=4704, arr_num_classes=arr_num_classes, num_attr=len(arr_num_classes), 
        device=device, scale=args.scale, fixed_means=False
    ).to(device)

# 3. Extractor
concept_extractor = FlowGMMConceptExtractor(flow_model, prior).to(device)

# 4. Predictor
c_dim = sum(arr_num_classes)
task_predictor = TaskPredictor(input_dim=c_dim, hidden_dim=128, num_classes=2).to(device)

# 5. CBM Wrapper
cbm = ModularCBM(concept_extractor, task_predictor).to(device)

# Optimization Setup
if OPTIMIZER == 'Adam':
    optimizer = optim.Adam(cbm.parameters(), lr=args.lr)
elif OPTIMIZER == 'SGD':
    optimizer = optim.SGD(cbm.parameters(), lr=args.lr, momentum=0.9)

criterion_y = nn.CrossEntropyLoss()

# Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
gpu_transform = K.RandomAffine(degrees=10, translate=(0.1, 0.1), p=1.0).to(device)

# State tracking
best_val_loss = float('inf')
best_val_acc_y = 0.0
reduction_count = 0
previous_lr = optimizer.param_groups[0]['lr']


# ==========================================
# 4. TRAINING LOOP
# ==========================================
logger.info("Starting Joint Training Loop...")

for epoch in range(args.epochs):
    cbm.train()
    
    # Trackers
    train_loss_total_sum = 0.0
    train_loss_c_sum = 0.0
    train_loss_y_sum = 0.0
    
    train_correct_c = 0 # All attributes match
    train_correct_y = 0
    train_total = 0
    
    for batch_X, batch_c, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
        batch_X, batch_c, batch_y = batch_X.to(device), batch_c.to(device), batch_y.to(device)
        
        # Transform
        if TRANSFORM > 0:
            n = batch_X.size(0)
            n_transform = int(TRANSFORM * n)
            if n_transform > 0:
                idx = torch.randperm(n, device=batch_X.device)[:n_transform]
                batch_X[idx] = gpu_transform(batch_X[idx])

        # Dequantization (Crucial for Flow)
        batch_X = (batch_X * 255. + torch.rand_like(batch_X)) / 256.
        batch_X = batch_X - 0.5
        if batch_X.dim() == 2: batch_X = batch_X.view(-1, 3, 28, 56)
        
        optimizer.zero_grad()
        
        # Forward Pass (Joint)
        # Returns: y_pred, c_continuous, preds, z, sldj, cf_outputs
        y_pred, _, c_preds, z, sldj, _ = cbm(batch_X, true_c_onehot=None, true_y=None, generate_cf=False)
        
        # Calculate Losses
        loss_c = prior.get_loss(z, sldj, batch_c) # NLL
        loss_y = criterion_y(y_pred, batch_y)     # CE
        
        total_loss = loss_y + (LAMBDA_C * loss_c)
        
        total_loss.backward()
        
        # Clip grads (Important for Flows + Joint training)
        torch.nn.utils.clip_grad_norm_(cbm.parameters(), 5)
        
        optimizer.step()
        
        # Metrics
        train_loss_total_sum += total_loss.item()
        train_loss_c_sum += loss_c.item()
        train_loss_y_sum += loss_y.item()
        
        train_correct_c += (c_preds == batch_c).all(dim=1).sum().item()
        train_correct_y += (y_pred.argmax(dim=1) == batch_y).sum().item()
        train_total += batch_y.size(0)
        
    # Averages
    avg_train_loss_total = train_loss_total_sum / len(train_loader)
    avg_train_loss_c = train_loss_c_sum / len(train_loader)
    avg_train_loss_y = train_loss_y_sum / len(train_loader)
    train_acc_c = train_correct_c / train_total
    train_acc_y = train_correct_y / train_total

    # ==========================================
    # VALIDATION
    # ==========================================
    cbm.eval()
    val_loss_total_sum = 0.0
    val_loss_c_sum = 0.0
    val_loss_y_sum = 0.0
    val_correct_c = 0
    val_correct_y = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_X, batch_c, batch_y in val_loader:
            batch_X, batch_c, batch_y = batch_X.to(device), batch_c.to(device), batch_y.to(device)
            
            # Dequantization
            batch_X = (batch_X * 255. + torch.rand_like(batch_X)) / 256.
            batch_X = batch_X - 0.5
            if batch_X.dim() == 2: batch_X = batch_X.view(-1, 3, 28, 56)

            y_pred, _, c_preds, z, sldj, _ = cbm(batch_X, true_c_onehot=None, true_y=None, generate_cf=False)
            
            loss_c = prior.get_loss(z, sldj, batch_c)
            loss_y = criterion_y(y_pred, batch_y)
            total_loss = loss_y + (LAMBDA_C * loss_c)
            
            val_loss_total_sum += total_loss.item()
            val_loss_c_sum += loss_c.item()
            val_loss_y_sum += loss_y.item()
            
            val_correct_c += (c_preds == batch_c).all(dim=1).sum().item()
            val_correct_y += (y_pred.argmax(dim=1) == batch_y).sum().item()
            val_total += batch_y.size(0)
            
    avg_val_loss_total = val_loss_total_sum / len(val_loader)
    avg_val_loss_c = val_loss_c_sum / len(val_loader)
    avg_val_loss_y = val_loss_y_sum / len(val_loader)
    val_acc_c = val_correct_c / val_total
    val_acc_y = val_correct_y / val_total

    # Scheduler Step
    scheduler.step(avg_train_loss_total)

    current_lr = optimizer.param_groups[0]['lr']
    if current_lr < previous_lr:
        reduction_count += 1
        previous_lr = current_lr
        logger.info(f"Reduction {reduction_count}: LR dropped to {current_lr}")
        
    if reduction_count >= 10:
        logger.info("Stopping: Too many LR reductions.")
        break

    # Logging
    logger.info(
        f'Epoch {epoch+1}/{args.epochs} | '
        f'T_Loss: {avg_train_loss_total:.4f} (C:{avg_train_loss_c:.2f} Y:{avg_train_loss_y:.4f}) | '
        f'T_Acc: C={train_acc_c:.4f} Y={train_acc_y:.4f} | '
        f'V_Acc: C={val_acc_c:.4f} Y={val_acc_y:.4f}'
    )
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow({
            'epoch': epoch + 1,
            'train_loss_total': avg_train_loss_total,
            'train_loss_c': avg_train_loss_c,
            'train_loss_y': avg_train_loss_y,
            'val_loss_total': avg_val_loss_total,
            'val_loss_c': avg_val_loss_c,
            'val_loss_y': avg_val_loss_y,
            'train_acc_c': train_acc_c,
            'train_acc_y': train_acc_y,
            'val_acc_c': val_acc_c,
            'val_acc_y': val_acc_y,
            'lr': current_lr
        })
    
    # Save Checkpoints
    if val_acc_y > best_val_acc_y:
        best_val_acc_y = val_acc_y
        torch.save({
            'cbm_state_dict': cbm.state_dict(),
            'epoch': epoch,
            'val_acc_y': val_acc_y
        }, os.path.join(model_save_dir, f"best_acc_{experiment_name}.pth"))

    if avg_val_loss_total < best_val_loss:
        best_val_loss = avg_val_loss_total
        torch.save({
            'cbm_state_dict': cbm.state_dict(),
            'epoch': epoch,
            'val_loss': avg_val_loss_total
        }, os.path.join(model_save_dir, f"best_loss_{experiment_name}.pth"))
        
    # Periodic Sampling
    if epoch % 10 == 0:
        save_samples_double_colored(
            cbm.concept_extractor.flow, prior, device, 
            num_attr=len(arr_num_classes), epoch=epoch, 
            save_dir=sample_save_dir, temp=0
        )

# Final Save
torch.save({
    'cbm_state_dict': cbm.state_dict(),
    'epoch': args.epochs
}, os.path.join(model_save_dir, f"final_{experiment_name}.pth"))