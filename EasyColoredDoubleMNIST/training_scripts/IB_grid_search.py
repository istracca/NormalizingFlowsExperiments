import subprocess
import itertools
import time
import sys
import os
import torch
from multiprocessing import Process

SCALES = [0.0]
MODELS = ['hybrid_v3_1x1_double']
PRIORS = ['CheckerboardIB_style']
OPTIMIZERS = ['Adam']
BETAS = [0.01,0.05,0.1,0.5,1.0]
TRANSFORMS = [0.5]
DROPOUT_P = [0.1]
VERSION = ['2_attr']
FIXED_MEANS = [False]
EPOCHS_WARMUP = [20]
STYLE_VARIANCE = [25.0]

def get_pending_experiments():
    """Generates combinations and filters out those that already have log files."""
    base_combos = list(itertools.product(SCALES, MODELS, PRIORS, OPTIMIZERS, BETAS, TRANSFORMS, DROPOUT_P, VERSION, FIXED_MEANS, EPOCHS_WARMUP))
    pending = []
    
    for combo in base_combos:
        scale, model, prior, opt, beta, trans, dropout, version, fix_m, epochs_warmup = combo
        
        if prior == 'CheckerboardIB_style':
            for style_variance in STYLE_VARIANCE:
                path = f"../experiments/logs/IB/{version}/IB_{scale}_{model}_{prior}_{beta}_{opt}_{trans}_{dropout}_{fix_m}_{epochs_warmup}_{style_variance}.log"
                if not os.path.exists(path):
                    pending.append((*combo, style_variance))
                else:
                    print(f"Skipping existing log: {path}")
        else:
            path = f"../experiments/logs/IB/{version}/IB_{scale}_{model}_{prior}_{beta}_{opt}_{trans}_{dropout}_{fix_m}_{epochs_warmup}.log"
            if not os.path.exists(path):
                pending.append((*combo, None))
            else:
                print(f"Skipping existing log: {path}")
            
    return pending

def run_worker(worker_id, gpu_id, experiments):
    if not experiments:
        print(f"Worker {worker_id} on GPU {gpu_id} has no experiments to run.")
        return

    print(f"Worker {worker_id} started on GPU {gpu_id} with {len(experiments)} tasks.")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    for i, (scale, model, prior, opt, beta, trans, dropout, version, fix_m, epochs_warmup, style_variance) in enumerate(experiments):
        print(f"\n[GPU {gpu_id} - Experiment {i+1}/{len(experiments)}]")
        print(f"Params: Scale={scale}, Model={model}, Prior={prior}, Beta={beta}, Opt={opt}, Transform={trans}, Dropout={dropout}, Version={version}, FixedMeans={fix_m}, EpochsWarmup={epochs_warmup}, StyleVariance={style_variance}")

        cmd = [
            sys.executable, "train_ib.py",
            "--scale", str(scale),
            "--model", model,
            "--optimizer", opt,
            "--prior", prior,
            "--beta", str(beta),
            "--transform", str(trans),
            "--dropout", str(dropout),
            "--version", version,
            "--epochs_warmup", str(epochs_warmup)
        ]
        
        if style_variance is not None:
            cmd.extend(["--style_variance", str(style_variance)])
            
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

    pending_combinations = get_pending_experiments()
    total_to_run = len(pending_combinations)

    if total_to_run == 0:
        print("All experiments already have log files. Nothing to run.")
        sys.exit(0)

    print(f"\n" + "=" * 30)
    print(f"Total experiments to run: {total_to_run}")
    print("=" * 30)

    try:
        JOBS_PER_GPU = int(input("Enter the number of parallel jobs to run per GPU: "))
        if JOBS_PER_GPU <= 0:
            print("Invalid number of jobs per GPU. Must be a positive integer.")
            sys.exit(1)
        elif JOBS_PER_GPU > 5:
            print("Warning: Setting a very high number of jobs per GPU may lead to out-of-memory errors.")
    except ValueError:
        print("Invalid input. Please enter an integer.")
        sys.exit(1)
        
    print(f"I will run {JOBS_PER_GPU} jobs in parallel on each GPU")

    physical_gpus = torch.cuda.device_count()
    total_workers = physical_gpus * JOBS_PER_GPU

    chunk_size = total_to_run // total_workers
    remainder = total_to_run % total_workers

    print("Job distribution per GPU:")
    for g in range(physical_gpus):
        gpu_task_count = 0
        for w in range(JOBS_PER_GPU):
            worker_idx = g * JOBS_PER_GPU + w
            count = chunk_size + (1 if worker_idx < remainder else 0)
            gpu_task_count += count
        print(f"  GPU {g}: {gpu_task_count} tasks")

    confirm = input("Proceed with running the experiments? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborting.")
        sys.exit(0)

    processes = []
    start_idx = 0
    start_time_global = time.time()

    for i in range(total_workers):
        assigned_gpu = i // JOBS_PER_GPU

        count = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + count

        if count > 0:
            gpu_experiments = pending_combinations[start_idx:end_idx]
            p = Process(target=run_worker, args=(i, assigned_gpu, gpu_experiments))
            p.start()
            processes.append(p)
            start_idx = end_idx

    for p in processes:
        p.join()

    total_duration = time.time() - start_time_global
    print("-" * 60)
    print(f"Grid Search Complete in {total_duration/3600:.2f} hours.")
