import subprocess
import itertools
import time
import sys
import os
import torch
from multiprocessing import Process

# --- Configuration ---
SCALES = [1, 3, 5]
BETAS = [0.25, 0.5, 1.0, 2.0, 4.0]
MODELS = ['hybrid', 'hybrid_v2']
OPTIMIZERS = ['Adam']
TRANSFORMS = [0.0, 0.5, 1.0]
DROPOUT_P = [0.0, 0.2]
FIXED_MEANS = [True, False]

JOBS_PER_GPU = 4

# Generate all combinations
combinations = list(itertools.product(SCALES, BETAS, MODELS, OPTIMIZERS, TRANSFORMS, DROPOUT_P, FIXED_MEANS))
total_runs = len(combinations)

def run_worker(worker_id, gpu_id, experiments):
    print(f"Worker {worker_id} started on GPU {gpu_id} with {len(experiments)} tasks.")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    for i, (scale, beta, model, opt, trans, dropout, fixed_means) in enumerate(experiments):
        print(f"\n[GPU {gpu_id} - Experiment {i+1}/{len(experiments)}]")
        print(f"Params: Scale={scale}, Beta={beta}, Model={model}, Opt={opt}, Transform={trans}, Dropout={dropout}, Fixed Means={fixed_means}")

        cmd = [
            sys.executable, "train_ib.py",
            "--scale", str(scale),
            "--beta", str(beta),
            "--model", model,
            "--optimizer", opt,
            "--transform", str(trans),
            "--dropout", str(dropout),
            "--fixed_means", str(fixed_means)
        ]

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