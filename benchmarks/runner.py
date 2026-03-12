"""Master benchmark runner for cross-species BPU experiments.

Orchestrates running all benchmarks across all connectomes with
biological and control (randomized) adjacency matrices. Generates
per-task and combined CSV result files.
"""

import argparse
import os
import sys
import time
import glob

import numpy as np
import scipy.sparse as sp
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.mnist import run_mnist
from benchmarks.fashion_mnist import run_fashion_mnist
from benchmarks.cifar10 import run_cifar10
from benchmarks.sequential_mnist import run_sequential_mnist
from benchmarks.audio import run_audio
from benchmarks.cartpole import run_cartpole


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY = {
    "MNIST": run_mnist,
    "FashionMNIST": run_fashion_mnist,
    "CIFAR10": run_cifar10,
    "SequentialMNIST": run_sequential_mnist,
    "Audio": run_audio,
    "CartPole": run_cartpole,
}


# ---------------------------------------------------------------------------
# Connectome loading
# ---------------------------------------------------------------------------

def _load_npz_adjacency(fpath):
    """Load adjacency matrix from an .npz file.

    Handles both dense numpy arrays and scipy sparse matrices stored
    using np.savez or scipy.sparse.save_npz.
    """
    try:
        # Try scipy sparse format first
        adj = sp.load_npz(fpath)
        return adj.toarray().astype(np.float32)
    except Exception:
        pass

    # Fall back to numpy npz
    data = np.load(fpath)
    if "adjacency" in data:
        return np.array(data["adjacency"], dtype=np.float32)
    # If only one array, use it
    keys = list(data.keys())
    if len(keys) == 1:
        return np.array(data[keys[0]], dtype=np.float32)
    raise KeyError(f"Cannot find adjacency matrix in {fpath}. Keys: {keys}")


def load_all_connectomes(organisms=None):
    """Load connectome adjacency matrices from data/processed/*.npz and
    data/subcircuits/*.npz.

    Args:
        organisms: Optional list of organism name substrings to filter by.

    Returns:
        List of dicts: [{name: str, adjacency: np.ndarray}, ...]
    """
    project_root = os.path.join(os.path.dirname(__file__), "..")
    search_dirs = [
        os.path.join(project_root, "data", "processed"),
        os.path.join(project_root, "data", "subcircuits"),
    ]

    connectomes = []
    for d in search_dirs:
        pattern = os.path.join(d, "*.npz")
        for fpath in sorted(glob.glob(pattern)):
            fname = os.path.splitext(os.path.basename(fpath))[0]
            try:
                adj = _load_npz_adjacency(fpath)
                if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
                    print(f"[runner] WARNING: {fpath} adjacency is not square "
                          f"({adj.shape}), skipping.")
                    continue
                connectomes.append({"name": fname, "adjacency": adj})
            except Exception as e:
                print(f"[runner] WARNING: Failed to load {fpath}: {e}")
                continue

    if organisms:
        connectomes = [
            c for c in connectomes
            if any(org.lower() in c["name"].lower() for org in organisms)
        ]

    if not connectomes:
        print("[runner] WARNING: No connectome .npz files found. "
              "Generating a small demo connectome (50 neurons).")
        demo_adj = _generate_random_adjacency(50, density=0.1, seed=0)
        connectomes.append({"name": "demo_random_50", "adjacency": demo_adj})

    return connectomes


# ---------------------------------------------------------------------------
# Control generation
# ---------------------------------------------------------------------------

def _generate_random_adjacency(n, density=None, seed=0):
    """Generate a random adjacency matrix with given density."""
    rng = np.random.RandomState(seed)
    if density is None:
        density = 0.1
    adj = (rng.rand(n, n) < density).astype(np.float32)
    np.fill_diagonal(adj, 0)
    return adj


def generate_controls(adjacency, name, n_controls=4, seed_base=1000):
    """Generate control adjacency matrices for comparison.

    Controls:
        1. Erdos-Renyi random: same size and density, random edges.
        2. Degree-preserved shuffle: randomly rewire edges preserving
           in/out degree sequence (approximate).
        3. Weight-shuffled: keep topology, randomly permute non-zero weights.
        4. Transposed: use A^T instead of A (reverses information flow).

    Args:
        adjacency: (N, N) numpy array. Original biological adjacency.
        name: Base name of the connectome.
        n_controls: Number of controls to generate (max 4).
        seed_base: Base seed for random number generators.

    Returns:
        List of dicts: [{name: str, adjacency: np.ndarray}, ...]
    """
    N = adjacency.shape[0]
    nonzero = np.count_nonzero(adjacency)
    density = nonzero / (N * N) if N > 0 else 0.1
    controls = []

    # Control 1: Erdos-Renyi random graph
    if n_controls >= 1:
        er_adj = _generate_random_adjacency(N, density=density, seed=seed_base)
        controls.append({
            "name": f"{name}_ctrl_erdos_renyi",
            "adjacency": er_adj,
        })

    # Control 2: Degree-preserved shuffle (configuration model approx.)
    if n_controls >= 2:
        rng = np.random.RandomState(seed_base + 1)
        shuffled = adjacency.copy()
        row_perm = rng.permutation(N)
        col_perm = rng.permutation(N)
        shuffled = shuffled[row_perm][:, col_perm]
        np.fill_diagonal(shuffled, 0)
        controls.append({
            "name": f"{name}_ctrl_degree_preserved",
            "adjacency": shuffled,
        })

    # Control 3: Weight-shuffled (topology fixed, weights permuted)
    if n_controls >= 3:
        rng = np.random.RandomState(seed_base + 2)
        ws_adj = adjacency.copy()
        mask = ws_adj != 0
        nonzero_vals = ws_adj[mask]
        rng.shuffle(nonzero_vals)
        ws_adj[mask] = nonzero_vals
        controls.append({
            "name": f"{name}_ctrl_weight_shuffled",
            "adjacency": ws_adj,
        })

    # Control 4: Transposed
    if n_controls >= 4:
        controls.append({
            "name": f"{name}_ctrl_transposed",
            "adjacency": adjacency.T.copy(),
        })

    return controls


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_full_experiment(tasks=None, device="cuda", seeds=None,
                        organisms=None, n_controls=4):
    """Run the full cross-species BPU benchmark suite.

    For each connectome:
        1. Run biological BPU on all tasks (multiple seeds each).
        2. Generate control adjacency matrices.
        3. Run each control on all tasks (multiple seeds each).
    Save results to results/{task}_results.csv and results/all_results.csv.

    Args:
        tasks: List of task names to run (default: all 6 tasks).
            Valid: MNIST, FashionMNIST, CIFAR10, SequentialMNIST, Audio, CartPole.
        device: Torch device string.
        seeds: List of random seeds (default: [42, 123, 456]).
        organisms: Optional list of organism name substrings to filter connectomes.
        n_controls: Number of control adjacency matrices per connectome (max 4).
    """
    if tasks is None:
        tasks = list(TASK_REGISTRY.keys())
    if seeds is None:
        seeds = [42, 123, 456]

    # Validate task names
    for t in tasks:
        if t not in TASK_REGISTRY:
            print(f"[runner] WARNING: Unknown task '{t}', skipping. "
                  f"Valid: {list(TASK_REGISTRY.keys())}")
    tasks = [t for t in tasks if t in TASK_REGISTRY]

    if not tasks:
        print("[runner] ERROR: No valid tasks specified.")
        return

    # Load connectomes
    print("=" * 70)
    print("Loading connectomes...")
    connectomes = load_all_connectomes(organisms=organisms)
    print(f"Loaded {len(connectomes)} connectome(s): "
          f"{[c['name'] for c in connectomes]}")

    # Build full run list: (config_dict, task_name, seed)
    run_list = []
    for conn in connectomes:
        bio_config = {
            "name": conn["name"],
            "adjacency": conn["adjacency"],
            "is_control": False,
        }
        for task in tasks:
            for seed in seeds:
                run_list.append((bio_config, task, seed))

        controls = generate_controls(
            conn["adjacency"], conn["name"],
            n_controls=n_controls, seed_base=1000
        )
        for ctrl in controls:
            ctrl_config = {
                "name": ctrl["name"],
                "adjacency": ctrl["adjacency"],
                "is_control": True,
            }
            for task in tasks:
                for seed in seeds:
                    run_list.append((ctrl_config, task, seed))

    total_runs = len(run_list)
    configs_per_conn = 1 + n_controls
    print(f"\nExperiment plan:")
    print(f"  Connectomes:     {len(connectomes)}")
    print(f"  Configs/conn:    {configs_per_conn} (1 bio + {n_controls} controls)")
    print(f"  Tasks:           {len(tasks)} {tasks}")
    print(f"  Seeds/config:    {len(seeds)} {seeds}")
    print(f"  Total runs:      {total_runs}")
    print("=" * 70)

    # Run experiments
    all_results = []
    t_start = time.time()

    pbar = tqdm(run_list, desc="Running experiments", unit="run")
    for i, (config, task_name, seed) in enumerate(pbar):
        pbar.set_postfix_str(
            f"{config['name'][:20]} | {task_name} | seed={seed}",
            refresh=True
        )

        task_fn = TASK_REGISTRY[task_name]

        try:
            result = task_fn(
                adjacency=config["adjacency"],
                name=config["name"],
                seed=seed,
                device=device,
            )
            result["is_control"] = config["is_control"]
            all_results.append(result)
        except Exception as e:
            print(f"\n[runner] ERROR on {config['name']}/{task_name}/seed={seed}: {e}")
            all_results.append({
                "name": config["name"],
                "task": task_name,
                "seed": seed,
                "is_control": config["is_control"],
                "error": str(e),
            })

        # Print ETA periodically
        elapsed = time.time() - t_start
        if i > 0 and (i + 1) % 5 == 0:
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (total_runs - i - 1)
            tqdm.write(
                f"  [{i+1}/{total_runs}] "
                f"Elapsed: {elapsed/60:.1f}min | "
                f"ETA: {remaining/60:.1f}min"
            )

    pbar.close()

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame(all_results)

    # Save combined results
    combined_path = os.path.join(results_dir, "all_results.csv")
    df.to_csv(combined_path, index=False)
    print(f"\nSaved combined results to {combined_path}")

    # Save per-task results
    for task_name in tasks:
        task_df = df[df["task"].str.contains(task_name, case=False, na=False)]
        if not task_df.empty:
            task_path = os.path.join(results_dir, f"{task_name}_results.csv")
            task_df.to_csv(task_path, index=False)
            print(f"Saved {task_name} results to {task_path}")

    # Print summary
    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"  Total time:  {total_time/60:.1f} minutes")
    print(f"  Total runs:  {len(all_results)}")
    if "accuracy" in df.columns:
        print(f"\nPer-task accuracy summary (mean across seeds):")
        summary = (
            df.dropna(subset=["accuracy"])
            .groupby(["task", "name"])["accuracy"]
            .mean()
            .unstack(level="name")
        )
        print(summary.to_string())
    print(f"{'=' * 70}")

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run BPU benchmark suite across biological connectomes."
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Tasks to run. Default: all. "
             f"Choices: {list(TASK_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device (default: cuda).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 123, 456],
        help="Random seeds for each run (default: 42 123 456).",
    )
    parser.add_argument(
        "--organisms", nargs="+", default=None,
        help="Filter connectomes by organism name substrings "
             "(e.g., celegans_herm ciona).",
    )
    parser.add_argument(
        "--n-controls", type=int, default=4,
        help="Number of control adjacency matrices per connectome (default: 4).",
    )

    args = parser.parse_args()

    # Validate device
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[runner] CUDA not available, falling back to CPU.")
        args.device = "cpu"
    else:
        if args.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"[runner] Using GPU: {gpu_name}")
            print(f"[runner] VRAM: {vram_gb:.1f} GB")

    run_full_experiment(
        tasks=args.tasks,
        device=args.device,
        seeds=args.seeds,
        organisms=args.organisms,
        n_controls=args.n_controls,
    )


if __name__ == "__main__":
    main()
