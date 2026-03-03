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

# --- IMPORTS ---
sys.path.append('../..') 
sys.path.append('../models')
sys.path.append('../priors')
from utils import set_seed
from cbm import FlowGMMConceptExtractor, TaskPredictor, ModularCBM
from hybrid_v3_1x1_double import GeneralFlow 
from CheckerboardGMM import CheckerboardGMM
from SimpleSplitGMM import SimpleSplitGMM

parser = argparse.ArgumentParser(description='Train CBM Sequential Paradigm (Cached)')
parser.add_argument('--scale', type=float, default=1.0, help='Scale parameter for the prior')
parser.add_argument('--model', type=str, default='hybrid_v3_1x1_double', help='Model name')
parser.add_argument('--prior', type=str, default='CheckerboardGMM', choices=['SimpleSplitGMM', 'CheckerboardGMM'], help='Prior type')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='Optimizer to use')
parser.add_argument('--transform', type=float, default=0.0, help='Percentage of data transformation to apply')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability for the model')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for MLP')
parser.add_argument('--epochs', type=int, default=50)
args = parser.parse_args()

module = importlib.import_module(args.model)
GeneralFlow = getattr(module, 'GeneralFlow')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(42)

# ==========================================
# 1. SETUP LOGGING
# ==========================================
experiment_name = f"CBM_seq_cached_{args.scale}_{args.model}_{args.prior}_{args.optimizer}_{args.transform}_{args.dropout}"
base_dir = '../experiments'

log_dir = f'{base_dir}/logs/CBM/sequential'
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

csv_dir = f'{base_dir}/csv/CBM/sequential'
os.makedirs(csv_dir, exist_ok=True)
csv_path = f'{csv_dir}/{experiment_name}.csv'

if os.path.exists(csv_path):
    os.remove(csv_path)

# Standardized Headers
headers = ['epoch', 
           'train_mlp_loss', 'train_mlp_acc', 'train_pipe_loss', 'train_pipe_acc', 
           'val_mlp_loss', 'val_mlp_acc', 'val_pipe_loss', 'val_pipe_acc', 
           'lr']
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()

model_save_dir = f'{base_dir}/models/CBM/sequential'
os.makedirs(model_save_dir, exist_ok=True)

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

# Temp loaders for caching (Sequential order)
train_loader_seq = DataLoader(TensorDataset(X_train, c_train, y_train), batch_size=128, shuffle=False)
val_loader_temp = DataLoader(TensorDataset(X_val, c_val, y_val), batch_size=128, shuffle=False)

# ==========================================
# 3. LOAD & FREEZE FLOW (X->C)
# ==========================================
pretrained_path = f"../../EasyColoredDoubleMNIST/experiments/models/GMM/4_attr/best_loss_{args.scale}_{args.model}_{args.prior}_{args.optimizer}_{args.transform}_{args.dropout}.pth"
logger.info(f"Loading legacy model from {pretrained_path}...")

arr_num_classes = [10, 10, 7, 7] 
flow_model = GeneralFlow(dropout_p=args.dropout).to(device)

if args.prior == 'CheckerboardGMM':
    gmm_prior = CheckerboardGMM(
        total_dim=4704, arr_num_classes=arr_num_classes, num_attr=len(arr_num_classes), 
        device=device, scale=1.0, fixed_means=False
    ).to(device)
elif args.prior == 'SimpleSplitGMM':
    from SimpleSplitGMM import SimpleSplitGMM
    gmm_prior = SimpleSplitGMM(
        total_dim=4704, arr_num_classes=arr_num_classes, num_attr=len(arr_num_classes), 
        device=device, scale=1.0, fixed_means=False
    ).to(device)

checkpoint = torch.load(pretrained_path, map_location=device)
flow_model.load_state_dict(checkpoint['model_state_dict'])
gmm_prior.load_state_dict(checkpoint['prior_state_dict'])

concept_extractor = FlowGMMConceptExtractor(flow_model, gmm_prior).to(device)
for param in concept_extractor.parameters():
    param.requires_grad = False
concept_extractor.eval() 
logger.info("Concept Extractor loaded and FROZEN.")

# ==========================================
# 4. PRE-COMPUTE / CACHE CONCEPTS
# ==========================================
logger.info("Pre-computing concepts (Raw Probabilities) for Train and Val...")

def get_one_hot(c_batch, arr_classes):
    """Helper to convert integer concepts to one-hot (CLEAN CONCEPTS)"""
    one_hot_list = []
    for i in range(len(arr_classes)):
        oh = torch.nn.functional.one_hot(c_batch[:, i], num_classes=arr_classes[i])
        one_hot_list.append(oh)
    return torch.cat(one_hot_list, dim=1).float()

def cache_data(loader):
    """Runs images through Flow ONCE and returns cached tensors"""
    c_pred_list = [] # Will hold RAW PROBABILITIES (Noisy)
    c_true_list = [] # Will hold Ground Truth Integers
    y_list = []
    
    with torch.no_grad():
        for batch_X, batch_c, batch_y in tqdm(loader, desc="Caching"):
            batch_X = batch_X.to(device)
            
            # Dequantize once
            batch_X = (batch_X * 255. + torch.rand_like(batch_X)) / 256.
            batch_X = batch_X - 0.5
            if batch_X.dim() == 2: batch_X = batch_X.view(-1, 3, 28, 56)
            
            # Flow Forward -> Returns RAW PROBABILITIES in c_continuous
            _, _, c_continuous, _ = concept_extractor(batch_X)
            
            c_pred_list.append(c_continuous) # Store Raw Probs
            c_true_list.append(batch_c)
            y_list.append(batch_y)
            
    return (torch.cat(c_pred_list).to(device), 
            torch.cat(c_true_list).to(device), 
            torch.cat(y_list).to(device))

# Cache Training Data (Used for Optimization AND Pipeline Eval)
train_c_pred_cached, _, train_y_cached = cache_data(train_loader_seq)

# Cache Validation Data (Used for Pipeline Eval & MLP Eval)
val_c_pred_cached, val_c_true_int, val_y_cached = cache_data(val_loader_temp)

# Pre-convert Val True concepts to one-hot (CLEAN) for diagnostic MLP eval
val_c_true_onehot = get_one_hot(val_c_true_int, arr_num_classes).to(device)

logger.info(f"Cached sizes: Train={train_c_pred_cached.shape}, Val={val_c_pred_cached.shape}")

# Create DataLoader for Optimization (Iterates over Noisy Concepts)
cached_train_loader = DataLoader(
    TensorDataset(train_c_pred_cached, train_y_cached), 
    batch_size=128, 
    shuffle=True
)


# ==========================================
# 5. INITIALIZE MLP & OPTIMIZER
# ==========================================
c_dim = sum(arr_num_classes)
num_y_classes = 2 

task_predictor = TaskPredictor(input_dim=c_dim, hidden_dim=128, num_classes=num_y_classes).to(device)
cbm = ModularCBM(concept_extractor, task_predictor).to(device)

optimizer_y = optim.Adam(task_predictor.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_y, mode='min', factor=0.5, patience=10, verbose=True)

best_val_loss = float('inf')
best_val_acc = 0.0
reduction_count = 0
previous_lr = optimizer_y.param_groups[0]['lr']


# ==========================================
# 6. TRAINING LOOP
# ==========================================
logger.info("Starting Ultra-Fast Training Loop (Sequential)...")

for epoch in range(args.epochs):
    
    # ----------------------------
    # A. OPTIMIZATION STEP (MLP on NOISY CONCEPTS)
    # ----------------------------
    # Sequential paradigm trains on the output of the Flow (c_pred)
    cbm.task_predictor.train() 
    
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_c_pred, batch_y in tqdm(cached_train_loader, desc=f"Epoch {epoch+1} [Optim]"):
        batch_c_pred, batch_y = batch_c_pred.to(device), batch_y.to(device)
        
        optimizer_y.zero_grad()
        
        # Forward (MLP sees Noisy Probabilities)
        y_pred = cbm.task_predictor(batch_c_pred)
        
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer_y.step()
        
        train_loss += loss.item()
        train_correct += (y_pred.argmax(dim=1) == batch_y).sum().item()
        train_total += batch_y.size(0)
        
    avg_train_loss = train_loss / len(cached_train_loader)
    avg_train_acc = train_correct / train_total

    # ----------------------------
    # B. TRAIN PIPELINE EVAL (Cached)
    # ----------------------------
    # In Sequential, this is mathematically identical to the optimization stats
    # (because we train on the pipeline output), but we track it for consistency.
    cbm.eval()
    with torch.no_grad():
        y_pred_pipe = cbm.task_predictor(train_c_pred_cached)
        loss_pipe = criterion(y_pred_pipe, train_y_cached)
        
        acc_pipe = (y_pred_pipe.argmax(dim=1) == train_y_cached).float().mean().item()
        avg_train_pipe_loss = loss_pipe.item()
        avg_train_pipe_acc = acc_pipe

    # ----------------------------
    # C. VALIDATION EVAL (Cached)
    # ----------------------------
    with torch.no_grad():
        # 1. Pipeline (Images -> Flow -> RAW PROBS -> MLP)
        # This is the "Real" accuracy of the model
        y_pred_val_pipe = cbm.task_predictor(val_c_pred_cached)
        loss_val_pipe = criterion(y_pred_val_pipe, val_y_cached)
        avg_val_pipe_acc = (y_pred_val_pipe.argmax(dim=1) == val_y_cached).float().mean().item()
        avg_val_pipe_loss = loss_val_pipe.item()

        # 2. MLP Only (True Concepts -> CLEAN ONE-HOT -> MLP)
        # Diagnostic: Did the MLP learn the logic abstractly?
        # Note: Since we trained on noise, it might perform worse on clean data!
        y_pred_val_mlp = cbm.task_predictor(val_c_true_onehot)
        loss_val_mlp = criterion(y_pred_val_mlp, val_y_cached)
        avg_val_mlp_acc = (y_pred_val_mlp.argmax(dim=1) == val_y_cached).float().mean().item()
        avg_val_mlp_loss = loss_val_mlp.item()

    # ----------------------------
    # D. SCHEDULER & LOGGING
    # ----------------------------
    scheduler.step(avg_train_loss) 

    current_lr = optimizer_y.param_groups[0]['lr']
    if current_lr < previous_lr:
        reduction_count += 1
        previous_lr = current_lr
        logger.info(f"Reduction {reduction_count}: LR dropped to {current_lr}")
        
    if reduction_count >= 10:
        logger.info("Stopping: Too many LR reductions.")
        break

    # Logging
    # Note: 'Trn MLP' here refers to the optimization loop (on noisy preds)
    logger.info(
        f'Epoch {epoch+1}/{args.epochs} | '
        f'Trn MLP: L={avg_train_loss:.4f} A={avg_train_acc:.4f} | '
        f'Trn Pipe: L={avg_train_pipe_loss:.4f} A={avg_train_pipe_acc:.4f} | '
        f'Val MLP: L={avg_val_mlp_loss:.4f} A={avg_val_mlp_acc:.4f} | '
        f'Val Pipe: L={avg_val_pipe_loss:.4f} A={avg_val_pipe_acc:.4f}'
    )
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow({
            'epoch': epoch + 1,
            'train_mlp_loss': avg_train_loss,
            'train_mlp_acc': avg_train_acc,
            'train_pipe_loss': avg_train_pipe_loss,
            'train_pipe_acc': avg_train_pipe_acc,
            'val_mlp_loss': avg_val_mlp_loss,
            'val_mlp_acc': avg_val_mlp_acc,
            'val_pipe_loss': avg_val_pipe_loss,
            'val_pipe_acc': avg_val_pipe_acc,
            'lr': current_lr
        })
    
    # Save Checkpoints
    if avg_val_pipe_acc > best_val_acc:
        best_val_acc = avg_val_pipe_acc
        torch.save({
            'task_predictor_state_dict': cbm.task_predictor.state_dict(),
            'epoch': epoch,
            'val_acc': avg_val_pipe_acc
        }, os.path.join(model_save_dir, f"best_acc_{experiment_name}.pth"))

    if avg_val_pipe_loss < best_val_loss:
        best_val_loss = avg_val_pipe_loss
        torch.save({
            'task_predictor_state_dict': cbm.task_predictor.state_dict(),
            'epoch': epoch,
            'val_loss': avg_val_pipe_loss
        }, os.path.join(model_save_dir, f"best_loss_{experiment_name}.pth"))

# Final Save
torch.save({
    'task_predictor_state_dict': cbm.task_predictor.state_dict(),
    'epoch': args.epochs
}, os.path.join(model_save_dir, f"final_{experiment_name}.pth"))