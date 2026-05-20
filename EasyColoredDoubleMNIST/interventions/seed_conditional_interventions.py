import os
import sys
import torch
import torch.nn.functional as F
import importlib
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from scipy import stats
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join('..', '..')))
sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))
from utils import set_seed

parser = argparse.ArgumentParser(description='Evaluate CC and Style on Colored MNIST (Conditional Paradigm).')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for dataloader')
args = parser.parse_args()

VERSION = '2_attr'
ARR_NUM_CLASSES = [10, 10]
NUM_ATTR = len(ARR_NUM_CLASSES)
COND_DIM = 64                                                                

GEN_CONFIG = {
    "MODEL": "conditional_scale",
    "DROPOUT": 0.1
}

MASTER_SEED = 42
rng = np.random.default_rng(MASTER_SEED)
SEEDS = ["original"] + rng.integers(0, 1000000, size=4).tolist()

MODELS_TO_LOAD = {}
for seed in SEEDS:
    if seed == "original":
        path = f"../experiments/models/Conditional/{VERSION}/best_loss_conditional_scale_GaussianPrior_Adam_0.5_0.1.pth"
        MODELS_TO_LOAD["Original"] = path
    else:
        path = f"../experiments_seed/models/Conditional/{VERSION}/best_loss_conditional_scale_GaussianPrior_Adam_0.5_0.1_{seed}.pth"
        MODELS_TO_LOAD[f"Seed_{seed}"] = path

DISC_CONFIG = {
    "MODEL": "disc_v3_double", 
    "PATH": f"../experiments/models/Disc/{VERSION}/best_loss_disc_v3_double_Adam_0.5_0.1.pth",
    "DROPOUT": 0.1
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_CSV = f"csv/Conditional_Colored_{VERSION}/cyc_cons_Conditional_aggregate.csv"

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
if os.path.exists(RESULTS_CSV):
    os.remove(RESULTS_CSV)

BASE_COLORS = torch.tensor([
    [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]
], dtype=torch.float32)

def get_confidence_interval(data_array, confidence=0.95):
    """Calculates mean and error margin for CI."""
    data_array = np.array(data_array, dtype=float)
    if np.isnan(data_array).all():
        return np.nan, 0.0
    if len(data_array) < 2:
        return np.nanmean(data_array), 0.0
    
    mean = np.nanmean(data_array)
    std_err = stats.sem(data_array, nan_policy='omit')
    if isinstance(std_err, np.ma.core.MaskedConstant) or np.isnan(std_err):
        return mean, 0.0
        
    h = std_err * stats.t.ppf((1 + confidence) / 2., len(data_array) - 1)
    return mean, h

def get_dominant_colors(imgs, device):
    imgs_shifted = imgs + 0.5 
    
    left_half = imgs_shifted[:, :, :, :28]
    right_half = imgs_shifted[:, :, :, 28:]
    
    mask_left = (left_half.max(dim=1, keepdim=True)[0] > 0.2).float()
    mask_right = (right_half.max(dim=1, keepdim=True)[0] > 0.2).float()
    
    mean_left = (left_half * mask_left).sum(dim=(2, 3)) / (mask_left.sum(dim=(2, 3)) + 1e-5)
    mean_right = (right_half * mask_right).sum(dim=(2, 3)) / (mask_right.sum(dim=(2, 3)) + 1e-5)
    
    base_colors = BASE_COLORS.to(device)
    
    color_left = torch.argmin(torch.cdist(mean_left, base_colors), dim=1)
    color_right = torch.argmin(torch.cdist(mean_right, base_colors), dim=1)
    
    return color_left, color_right

def evaluate_single_model(gen, disc, loader, device, model_name):
    gen.eval()
    disc.eval()
    
    stats_tracker = {
        "attr0": {"cc_tgt": 0, "cc_non_tgt": 0, "cc_overall": 0, "style_tgt": 0, "style_non_tgt": 0, "style_overall": 0, "total": 0},
        "attr1": {"cc_tgt": 0, "cc_non_tgt": 0, "cc_overall": 0, "style_tgt": 0, "style_non_tgt": 0, "style_overall": 0, "total": 0},
        "both":  {"cc_tgt": 0, "cc_non_tgt": 0, "cc_overall": 0, "style_tgt": 0, "style_non_tgt": 0, "style_overall": 0, "total": 0}
    }

    with torch.no_grad():
        for batch_X, batch_y in tqdm(loader, desc=f"Evaluating {model_name}", leave=False):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            B = batch_X.size(0)
            
            batch_X = (batch_X * 255.0 + torch.rand_like(batch_X)) / 256.0
            batch_X_centered = batch_X - 0.5
            
            orig_color_left, orig_color_right = get_dominant_colors(batch_X_centered, device)
            
            logits_pre_tuple = disc(batch_X_centered)
            if not isinstance(logits_pre_tuple, (list, tuple)): logits_pre_tuple = (logits_pre_tuple,)
            preds_pre = torch.stack([torch.argmax(logits, dim=1) for logits in logits_pre_tuple], dim=1)
            
            y_cond_pre = torch.cat([
                F.one_hot(preds_pre[:, 0], num_classes=10),
                F.one_hot(preds_pre[:, 1], num_classes=10)
            ], dim=1).float()
            z, _ = gen(batch_X_centered, y_cond_pre)
            
            for offset0 in range(10):
                for offset1 in range(10):
                    if offset0 == 0 and offset1 == 0: continue
                    
                    t0 = (preds_pre[:, 0] + offset0) % 10
                    t1 = (preds_pre[:, 1] + offset1) % 10
                    
                    if offset0 != 0 and offset1 == 0: int_type = "attr0"
                    elif offset0 == 0 and offset1 != 0: int_type = "attr1"
                    else: int_type = "both"
                    
                    y_cond_target = torch.cat([
                        F.one_hot(t0, num_classes=10),
                        F.one_hot(t1, num_classes=10)
                    ], dim=1).float()
                    
                    img_interv = gen.inverse(z, y_cond_target)
                    
                    logits_post_tuple = disc(img_interv)
                    if not isinstance(logits_post_tuple, (list, tuple)): logits_post_tuple = (logits_post_tuple,)
                    preds_post = torch.stack([torch.argmax(logits, dim=1) for logits in logits_post_tuple], dim=1)
                    
                    col_left, col_right = get_dominant_colors(img_interv, device)
                    
                    correct_attr0 = (preds_post[:, 0] == t0)
                    correct_attr1 = (preds_post[:, 1] == t1)
                    style_match_0 = (col_left == orig_color_left)
                    style_match_1 = (col_right == orig_color_right)
                    
                    cc_overall_match = (correct_attr0 & correct_attr1)
                    
                    stats_tracker[int_type]["style_overall"] += (style_match_0 & style_match_1).sum().item()
                    stats_tracker[int_type]["cc_overall"] += cc_overall_match.sum().item()

                    if int_type == "attr0":
                        stats_tracker[int_type]["cc_tgt"] += correct_attr0.sum().item()
                        stats_tracker[int_type]["cc_non_tgt"] += correct_attr1.sum().item()
                        stats_tracker[int_type]["style_tgt"] += style_match_0.sum().item()
                        stats_tracker[int_type]["style_non_tgt"] += style_match_1.sum().item()
                    elif int_type == "attr1":
                        stats_tracker[int_type]["cc_tgt"] += correct_attr1.sum().item()
                        stats_tracker[int_type]["cc_non_tgt"] += correct_attr0.sum().item()
                        stats_tracker[int_type]["style_tgt"] += style_match_1.sum().item()
                        stats_tracker[int_type]["style_non_tgt"] += style_match_0.sum().item()
                    elif int_type == "both":
                        stats_tracker[int_type]["cc_tgt"] += cc_overall_match.sum().item()                             
                        stats_tracker[int_type]["style_tgt"] += (style_match_0 & style_match_1).sum().item()
                        
                    stats_tracker[int_type]["total"] += B

    results = []
    
    for int_type in ["attr0", "attr1", "both"]:
        s = stats_tracker[int_type]
        t_total = s["total"]
        num_interventions = 1 if int_type in ["attr0", "attr1"] else 2
        
        results.append({
            'Model': model_name,
            'intervention_group': int_type,
            'concepts_intervened': num_interventions,
            'cc_target': s["cc_tgt"] / t_total if t_total > 0 else np.nan,
            'cc_non_target': s["cc_non_tgt"] / t_total if t_total > 0 and int_type != "both" else np.nan,
            'cc_overall': s["cc_overall"] / t_total if t_total > 0 else np.nan,
            'style_target': s["style_tgt"] / t_total if t_total > 0 else np.nan,
            'style_non_target': s["style_non_tgt"] / t_total if t_total > 0 and int_type != "both" else np.nan,
            'style_overall': s["style_overall"] / t_total if t_total > 0 else np.nan,
            'total_samples': t_total
        })
        
    tot_1_cc_tgt = stats_tracker["attr0"]["cc_tgt"] + stats_tracker["attr1"]["cc_tgt"]
    tot_1_cc_ntgt = stats_tracker["attr0"]["cc_non_tgt"] + stats_tracker["attr1"]["cc_non_tgt"]
    tot_1_cc_ovr = stats_tracker["attr0"]["cc_overall"] + stats_tracker["attr1"]["cc_overall"]
    tot_1_st_tgt = stats_tracker["attr0"]["style_tgt"] + stats_tracker["attr1"]["style_tgt"]
    tot_1_st_ntgt = stats_tracker["attr0"]["style_non_tgt"] + stats_tracker["attr1"]["style_non_tgt"]
    tot_1_st_ovr = stats_tracker["attr0"]["style_overall"] + stats_tracker["attr1"]["style_overall"]
    tot_1_total = stats_tracker["attr0"]["total"] + stats_tracker["attr1"]["total"]
    
    results.append({
        'Model': model_name,
        'intervention_group': 'AGGREGATE_1_INTERVENTION',
        'concepts_intervened': 1,
        'cc_target': tot_1_cc_tgt / tot_1_total if tot_1_total > 0 else np.nan,
        'cc_non_target': tot_1_cc_ntgt / tot_1_total if tot_1_total > 0 else np.nan,
        'cc_overall': tot_1_cc_ovr / tot_1_total if tot_1_total > 0 else np.nan,
        'style_target': tot_1_st_tgt / tot_1_total if tot_1_total > 0 else np.nan,
        'style_non_target': tot_1_st_ntgt / tot_1_total if tot_1_total > 0 else np.nan,
        'style_overall': tot_1_st_ovr / tot_1_total if tot_1_total > 0 else np.nan,
        'total_samples': tot_1_total
    })

    tot_all_cc_tgt = tot_1_cc_tgt + stats_tracker["both"]["cc_tgt"]
    tot_all_cc_ovr = tot_1_cc_ovr + stats_tracker["both"]["cc_overall"]
    tot_all_st_tgt = tot_1_st_tgt + stats_tracker["both"]["style_tgt"]
    tot_all_st_ovr = tot_1_st_ovr + stats_tracker["both"]["style_overall"]
    tot_all_total = tot_1_total + stats_tracker["both"]["total"]
    
    results.append({
        'Model': model_name,
        'intervention_group': 'ALL',
        'concepts_intervened': 'ALL',
        'cc_target': tot_all_cc_tgt / tot_all_total if tot_all_total > 0 else np.nan,
        'cc_non_target': tot_1_cc_ntgt / tot_1_total if tot_1_total > 0 else np.nan,                    
        'cc_overall': tot_all_cc_ovr / tot_all_total if tot_all_total > 0 else np.nan,
        'style_target': tot_all_st_tgt / tot_all_total if tot_all_total > 0 else np.nan,
        'style_non_target': tot_1_st_ntgt / tot_1_total if tot_1_total > 0 else np.nan,
        'style_overall': tot_all_st_ovr / tot_all_total if tot_all_total > 0 else np.nan,
        'total_samples': tot_all_total
    })

    return results


if __name__ == "__main__":
    set_seed(42)
    
    print("Loading Discriminator...")
    disc_module = importlib.import_module(DISC_CONFIG['MODEL'])
    PseudoResNet = getattr(disc_module, 'PseudoResNet')
    disc_model = PseudoResNet(arr_num_classes=ARR_NUM_CLASSES, in_channels=3, dropout_p=DISC_CONFIG['DROPOUT'], device=DEVICE).to(DEVICE)
    disc_model.load_state_dict(torch.load(DISC_CONFIG['PATH'], map_location=DEVICE)['model_state_dict'])
    
    print("Loading Conditional Generator Architecture...")
    gen_module = importlib.import_module(GEN_CONFIG['MODEL'])
    GeneralFlow = getattr(gen_module, 'GeneralFlow')
    gen = GeneralFlow(num_classes=sum(ARR_NUM_CLASSES), dropout_p=GEN_CONFIG['DROPOUT'], cond_dim=COND_DIM).to(DEVICE)
    
    print("Loading test dataset...")
    data = np.load('../data/easy_colored_double_mnist.npz')
    X_test, y_test = data['X_test'], data['y_test']
    
    y_test = y_test[:, 0:2]
    
    X_test_tensor = torch.tensor(X_test.transpose(0, 3, 1, 2), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    if len(test_loader) > 0:
        all_model_results = []
        
        for model_name, model_path in MODELS_TO_LOAD.items():
            if not os.path.exists(model_path):
                print(f"Skipping {model_name}: Path not found ({model_path})")
                continue
                
            print(f"\nLoading weights for {model_name}...")
            checkpoint = torch.load(model_path, map_location=DEVICE)
            gen.load_state_dict(checkpoint['model_state_dict'])
            
            res = evaluate_single_model(gen, disc_model, test_loader, DEVICE, model_name)
            all_model_results.extend(res)

        if len(all_model_results) == 0:
            print("No models evaluated. Exiting.")
            sys.exit(0)

        df_flat = pd.DataFrame(all_model_results)
        
        metrics_to_agg = ['cc_target', 'cc_non_target', 'cc_overall', 'style_target', 'style_non_target', 'style_overall']
                          
        pivot_df = df_flat.pivot(index=['intervention_group', 'concepts_intervened', 'total_samples'], 
                                 columns='Model', 
                                 values=metrics_to_agg)
        
        pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]
        pivot_df = pivot_df.reset_index()

        for m in metrics_to_agg:
            model_cols = [col for col in pivot_df.columns if col.startswith(f"{m}_")]
            
            def apply_ci(row):
                data = row[model_cols].values.astype(float)
                mean, ci = get_confidence_interval(data, confidence=0.95)
                return pd.Series([mean, ci])
                
            pivot_df[[f"{m}_mean", f"{m}_ci_95"]] = pivot_df.apply(apply_ci, axis=1)

        pivot_df.to_csv(RESULTS_CSV, index=False)
        print(f"\nEvaluation complete. Aggregated metrics saved to '{RESULTS_CSV}'.")