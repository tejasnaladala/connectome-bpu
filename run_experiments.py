"""Master experiment runner — runs all BPU experiments with incremental saving.

Usage:
    python run_experiments.py                  # Full run
    python run_experiments.py --quick          # Quick mode (small organisms only, MNIST only, 5 epochs)
    python run_experiments.py --task MNIST     # Single task
    python run_experiments.py --organism ciona # Single organism
"""
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import load_npz
import os, sys, csv, json, glob, time, argparse, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.bpu import BPU, SequentialBPU
from models.controls import generate_all_controls
from benchmarks.mnist import run_mnist
from benchmarks.fashion_mnist import run_fashion_mnist
from benchmarks.cifar10 import run_cifar10
from benchmarks.sequential_mnist import run_sequential_mnist
from benchmarks.audio import run_audio
from benchmarks.cartpole import run_cartpole

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(RESULTS_DIR, "all_results.csv")

# Benchmark config
BENCHMARKS = {
    "MNIST": {"func": run_mnist, "epochs": 20, "batch_size": 128},
    "FashionMNIST": {"func": run_fashion_mnist, "epochs": 20, "batch_size": 128},
    "CIFAR10": {"func": run_cifar10, "epochs": 30, "batch_size": 128},
    "SequentialMNIST": {"func": run_sequential_mnist, "epochs": 10, "batch_size": 64},
    "Audio": {"func": run_audio, "epochs": 15, "batch_size": 64},
    "CartPole": {"func": run_cartpole, "epochs": 300, "batch_size": 1},
}

SEEDS = [42, 43, 44]


def load_existing_results():
    """Load previously completed results to avoid re-running."""
    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
        completed = set()
        for _, row in df.iterrows():
            key = (row["name"], row["task"], int(row["seed"]))
            completed.add(key)
        return df.to_dict("records"), completed
    return [], set()


def save_results(results):
    """Save all results to CSV."""
    if not results:
        return
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)


def get_device():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return "cuda"
    print("WARNING: No GPU available, using CPU (will be slow)")
    return "cpu"


def load_all_connectomes():
    """Load all biological connectomes."""
    base = os.path.dirname(os.path.abspath(__file__))
    connectomes = {}

    # Whole organisms
    for f in sorted(glob.glob(os.path.join(base, "data/processed/*_adjacency.npz"))):
        name = os.path.basename(f).replace("_adjacency.npz", "")
        connectomes[name] = {"adj": load_npz(f), "type": "whole_organism"}

    # Sub-circuits
    for f in sorted(glob.glob(os.path.join(base, "data/subcircuits/*_adjacency.npz"))):
        name = "adult_" + os.path.basename(f).replace("_adjacency.npz", "")
        connectomes[name] = {"adj": load_npz(f), "type": "subcircuit"}

    return connectomes


def check_vram_fit(N, task, device):
    """Check if a model will fit in VRAM."""
    if device == "cpu":
        return True, 128

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    # Estimate memory: W_rec (N²×4) + input_proj + output_proj + gradients + batch
    d_in = {"MNIST": 784, "FashionMNIST": 784, "CIFAR10": 3072,
            "SequentialMNIST": 1, "Audio": 400, "CartPole": 4}.get(task, 784)

    model_mb = (N * N * 4 + d_in * N * 4 + N * 10 * 4) / 1e6
    # Factor 3x for gradients + activations + optimizer state
    total_mb = model_mb * 3

    if total_mb > vram_gb * 1000 * 0.8:  # 80% threshold
        return False, 0

    # Adaptive batch size
    if N > 3000:
        batch_size = 32
    elif N > 1500:
        batch_size = 64
    else:
        batch_size = 128

    return True, batch_size


def run_single_experiment(adj, name, organism, bpu_type, task_name, task_config,
                          device, seed, batch_size_override=None):
    """Run a single experiment and return result dict."""
    func = task_config["func"]
    epochs = task_config["epochs"]
    batch_size = batch_size_override or task_config.get("batch_size", 128)

    kwargs = {
        "adjacency": adj,
        "name": name,
        "epochs": epochs,
        "device": device,
        "seed": seed,
    }

    # CartPole uses n_episodes instead of epochs
    if task_name == "CartPole":
        kwargs["n_episodes"] = epochs
        del kwargs["epochs"]
    else:
        kwargs["batch_size"] = batch_size
        kwargs["lr"] = 1e-3

    result = func(**kwargs)
    result["organism"] = organism
    result["type"] = bpu_type
    return result


def main():
    parser = argparse.ArgumentParser(description="Run BPU experiments")
    parser.add_argument("--quick", action="store_true", help="Quick mode: small nets, MNIST, 5 epochs")
    parser.add_argument("--task", type=str, default=None, help="Run single task")
    parser.add_argument("--organism", type=str, default=None, help="Run single organism")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="Seeds to use")
    parser.add_argument("--no-controls", action="store_true", help="Skip control experiments")
    parser.add_argument("--bio-only", action="store_true", help="Only biological connectomes")
    args = parser.parse_args()

    device = get_device()

    print("\n" + "=" * 70)
    print("CROSS-SPECIES BPU EXPERIMENT BATTERY")
    print("=" * 70)

    # Load connectomes
    connectomes = load_all_connectomes()
    print(f"\nLoaded {len(connectomes)} biological connectomes:")
    for name, info in sorted(connectomes.items(), key=lambda x: x[1]["adj"].shape[0]):
        adj = info["adj"]
        print(f"  {name:<30} {adj.shape[0]:>5} neurons  {adj.nnz:>7} edges  ({info['type']})")

    # Filter if requested
    if args.organism:
        connectomes = {k: v for k, v in connectomes.items() if args.organism in k}
        print(f"\nFiltered to organism: {args.organism} ({len(connectomes)} networks)")

    # Select tasks
    if args.quick:
        tasks = {"MNIST": {**BENCHMARKS["MNIST"], "epochs": 5}}
        seeds = [42]
        # Only small connectomes in quick mode
        connectomes = {k: v for k, v in connectomes.items()
                       if v["adj"].shape[0] <= 600}
    elif args.task:
        tasks = {args.task: BENCHMARKS[args.task]}
        seeds = args.seeds
    else:
        tasks = BENCHMARKS
        seeds = args.seeds

    # Load existing results
    all_results, completed = load_existing_results()
    print(f"\nPreviously completed: {len(completed)} experiments")

    # Count total experiments
    n_bio = len(connectomes) * len(tasks) * len(seeds)
    n_ctrl = n_bio * 4 if not args.no_controls and not args.bio_only else 0
    n_total = n_bio + n_ctrl - len(completed)
    print(f"Remaining experiments: {n_total}")
    print(f"Tasks: {list(tasks.keys())}")
    print(f"Seeds: {seeds}")

    t_start = time.time()
    n_done = 0

    # Run experiments
    for org_name, org_info in sorted(connectomes.items(), key=lambda x: x[1]["adj"].shape[0]):
        adj = org_info["adj"]
        N = adj.shape[0]

        print(f"\n{'='*60}")
        print(f"ORGANISM: {org_name} ({N} neurons)")
        print(f"{'='*60}")

        # Generate controls once per organism
        if not args.no_controls and not args.bio_only:
            print("Generating controls...")
            controls = generate_all_controls(adj, seed=42)
        else:
            controls = {}

        for task_name, task_config in tasks.items():
            fits, batch_size = check_vram_fit(N, task_name, device)
            if not fits:
                print(f"\n  SKIP {task_name}: model too large for VRAM ({N} neurons)")
                continue

            # Skip SequentialMNIST for large networks (too slow)
            if task_name == "SequentialMNIST" and N > 1500:
                print(f"\n  SKIP {task_name}: too slow for {N}-neuron networks")
                continue

            for seed in seeds:
                # === Biological ===
                key = (org_name, task_name, seed)
                if key not in completed:
                    try:
                        print(f"\n  [{n_done+1}] {org_name} / {task_name} / bio / seed={seed}")
                        result = run_single_experiment(
                            adj, org_name, org_name, "biological",
                            task_name, task_config, device, seed, batch_size
                        )
                        all_results.append(result)
                        completed.add(key)
                        n_done += 1
                        print(f"    -> accuracy={result['accuracy']:.4f}, time={result.get('train_time_sec', 0)}s")
                        save_results(all_results)
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"    ERROR: {e}")
                        traceback.print_exc()

                # === Controls ===
                if not args.no_controls and not args.bio_only:
                    for ctrl_name, ctrl_adj in controls.items():
                        ctrl_full_name = f"{org_name}_{ctrl_name}"
                        key = (ctrl_full_name, task_name, seed)
                        if key not in completed:
                            try:
                                print(f"  [{n_done+1}] {ctrl_full_name} / {task_name} / seed={seed}")
                                result = run_single_experiment(
                                    ctrl_adj, ctrl_full_name, org_name, ctrl_name,
                                    task_name, task_config, device, seed, batch_size
                                )
                                all_results.append(result)
                                completed.add(key)
                                n_done += 1
                                print(f"    -> accuracy={result['accuracy']:.4f}")
                                save_results(all_results)
                                torch.cuda.empty_cache()
                            except Exception as e:
                                print(f"    ERROR: {e}")

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"COMPLETED: {n_done} experiments in {elapsed/60:.1f} minutes")
    print(f"Total results: {len(all_results)}")
    print(f"Saved to: {RESULTS_CSV}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
