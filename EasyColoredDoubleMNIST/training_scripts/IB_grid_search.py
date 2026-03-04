import subprocess
import itertools
import time
import sys
import os
import torch
from multiprocessing import Process

# --- Configuration ---
SCALES = [1.0]
MODELS = ['hybrid_v3_1x1_double']
PRIORS = ['CheckerboardIB', 'CheckerboardIB_style']
OPTIMIZERS = ['Adam']
BETAS = [0.001, 0.01, 0.1]
TRANSFORMS = [0.25, 0.5, 0.75]
DROPOUT_P = [0.1, 0.2, 0.3]
VERSION = ['2_attr','4_attr']
FIXED_MEANS = [False]

JOBS_PER_GPU = 3

# Generate all combinations
combinations = list(itertools.product(SCALES, MODELS, PRIORS, OPTIMIZERS, BETAS, TRANSFORMS, DROPOUT_P, VERSION, FIXED_MEANS))
total_runs = len(combinations)

def run_worker(worker_id, gpu_id, experiments):
    print(f"Worker {worker_id} started on GPU {gpu_id} with {len(experiments)} tasks.")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    for i, (scale, model, prior, opt, beta, trans, dropout, version, fix_m) in enumerate(experiments):
        path = f"../experiments/logs/IB/{version}/IB_{scale}_{model}_{prior}_{beta}_{opt}_{trans}_{dropout}_False.log"
        if os.path.exists(path):
            print(f"Skipping existing log: {path}")
            continue
        print(f"\n[GPU {gpu_id} - Experiment {i+1}/{len(experiments)}]")
        print(f"Params: Scale={scale}, Model={model}, Prior={prior}, Beta={beta}, Opt={opt}, Transform={trans}, Dropout={dropout}, Version={version}")

        cmd = [
            sys.executable, "train_ib.py",
            "--scale", str(scale),
            "--model", model,
            "--optimizer", opt,
            "--prior", prior,
            "--beta", str(beta),
            "--transform", str(trans),
            "--dropout", str(dropout),
            "--version", version
        ]
        if fix_m:
            cmd.append("--fixed_means")

        try:
            subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"Worker {worker_id} failed with error code {e.returncode}")
        except KeyboardInterrupt:
            print(f"\nWorker {worker_id} on GPU {gpu_id} interrupted by user.")
            break

        print(f"Worker {worker_id} finished experiment on GPU {gpu_id}.")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("No CUDA-capable GPU detected.")

    physical_gpus = torch.cuda.device_count()
    total_workers = physical_gpus * JOBS_PER_GPU
    
    print(f"Detected {physical_gpus} physical GPUs with {JOBS_PER_GPU} jobs each, totaling {total_workers} workers.")

    chunk_size = len(combinations) // total_workers
    remainder = len(combinations) % total_workers

    processes = []
    start_idx = 0

    start_time_global = time.time()

    for i in range(total_workers):
        assigned_gpu = i // JOBS_PER_GPU

        count = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + count

        if count > 0:
            gpu_experiments = combinations[start_idx:end_idx]
            p = Process(target=run_worker, args=(i, assigned_gpu, gpu_experiments))
            p.start()
            processes.append(p)
            start_idx = end_idx

    for p in processes:
        p.join()

    total_duration = time.time() - start_time_global
    print("-" * 60)
    print(f"Grid Search Complete in {total_duration/3600:.2f} hours.")