#!/usr/bin/env python3
"""Master experiment runner for the cross-species BPU research project.

Handles the ENTIRE pipeline from raw connectome data to final analysis results.

Phases:
  1. Standardize all connectomes to sparse adjacency matrices
  2. Generate synthetic control graphs for each connectome
  3. Compute graph-theoretic metrics for all networks
  4. Run all benchmark tasks on all networks
  5. Run statistical analysis (scaling laws, correlations, specialization)

Usage:
  python run_all.py --phase all          # Run everything
  python run_all.py --phase 1            # Only standardize connectomes
  python run_all.py --phase 4            # Only run benchmarks
  python run_all.py --phase 4 5          # Benchmarks + analysis
  python run_all.py --phase all --small  # Quick test (1 connectome, 1 task, 1 seed)
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Project root (everything is relative to this)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SUBCIRCUIT_DIR = os.path.join(PROJECT_ROOT, "data", "subcircuits")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

for _d in [PROCESSED_DIR, SUBCIRCUIT_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Optional tqdm
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        total = kwargs.get("total", None)
        for i, item in enumerate(iterable):
            if total:
                print(f"\r  {desc} [{i+1}/{total}]", end="", flush=True)
            yield item
        if total:
            print()

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
_T0 = time.time()


def log(msg, level="INFO"):
    elapsed = time.time() - _T0
    h, remainder = divmod(int(elapsed), 3600)
    m, s = divmod(remainder, 60)
    ts = f"{h:02d}:{m:02d}:{s:02d}"
    print(f"[{ts}] [{level}] {msg}", flush=True)


def log_phase(phase_num, title):
    print(f"\n{'='*72}", flush=True)
    log(f"PHASE {phase_num}: {title}")
    print(f"{'='*72}", flush=True)


# ============================================================================
# PHASE 1: STANDARDIZE ALL CONNECTOMES
# ============================================================================

def _edge_list_to_sparse(csv_path):
    """Read an edge list CSV (Source, Target, Weight) and return sparse matrix + metadata."""
    df = pd.read_csv(csv_path)
    sources = df["Source"].astype(str).values
    targets = df["Target"].astype(str).values
    weights = df["Weight"].astype(float).values

    # Build node mapping
    all_nodes = sorted(set(sources) | set(targets))
    node_to_id = {n: i for i, n in enumerate(all_nodes)}
    N = len(all_nodes)

    row_ids = np.array([node_to_id[s] for s in sources])
    col_ids = np.array([node_to_id[t] for t in targets])

    adj = sp.csr_matrix((weights, (row_ids, col_ids)), shape=(N, N))

    metadata = {
        "n_neurons": N,
        "n_edges": int(adj.nnz),
        "node_names": all_nodes,
        "source_file": os.path.basename(csv_path),
    }
    return adj, metadata


def standardize_edge_list(csv_path, name, output_dir=PROCESSED_DIR):
    """Standardize an edge-list CSV connectome and save as .npz + .json."""
    adj_path = os.path.join(output_dir, f"{name}_adjacency.npz")
    meta_path = os.path.join(output_dir, f"{name}_metadata.json")

    if os.path.exists(adj_path):
        log(f"  {name}: already exists, skipping.")
        return sp.load_npz(adj_path)

    log(f"  {name}: reading {csv_path}")
    adj, metadata = _edge_list_to_sparse(csv_path)
    sp.save_npz(adj_path, adj)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"  {name}: saved {metadata['n_neurons']} neurons, {metadata['n_edges']} edges")
    return adj


def standardize_drosophila_larva():
    """Read the full dense connectivity matrix for Drosophila larva.

    The CSV has neuron IDs as the first row and first column.
    First cell (row 0, col 0) is empty or an index marker.
    """
    name = "drosophila_larva"
    adj_path = os.path.join(PROCESSED_DIR, f"{name}_adjacency.npz")
    meta_path = os.path.join(PROCESSED_DIR, f"{name}_metadata.json")

    if os.path.exists(adj_path):
        log(f"  {name}: already exists, skipping.")
        return sp.load_npz(adj_path)

    csv_path = os.path.join(
        RAW_DIR, "drosophila_larva", "Supplementary-Data-S1",
        "all-all_connectivity_matrix.csv"
    )
    log(f"  {name}: reading dense connectivity matrix (this may take a moment)...")

    # Read the full CSV. First column = neuron IDs (index), first row = neuron IDs (header).
    df = pd.read_csv(csv_path, index_col=0)
    neuron_ids = [str(x) for x in df.columns]
    N = len(neuron_ids)
    log(f"  {name}: parsed {N}x{N} matrix")

    # Convert to sparse (float64)
    matrix = df.values.astype(np.float64)
    adj = sp.csr_matrix(matrix)

    sp.save_npz(adj_path, adj)
    metadata = {
        "n_neurons": N,
        "n_edges": int(adj.nnz),
        "node_names": neuron_ids,
        "source_file": "all-all_connectivity_matrix.csv",
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"  {name}: saved {N} neurons, {adj.nnz} edges")
    return adj


def standardize_drosophila_adult_subcircuits():
    """Extract sub-circuit adjacency matrices from the adult Drosophila connectome.

    For each neuropil region, find neurons with majority presence in that region,
    then extract the subgraph of connections among those neurons.
    """
    connections_path = os.path.join(
        RAW_DIR, "drosophila_adult", "proofread_connections_783.feather"
    )
    pre_counts_path = os.path.join(
        RAW_DIR, "drosophila_adult", "per_neuron_neuropil_count_pre_783.feather"
    )
    post_counts_path = os.path.join(
        RAW_DIR, "drosophila_adult", "per_neuron_neuropil_count_post_783.feather"
    )

    # Check all files exist
    for p in [connections_path, pre_counts_path, post_counts_path]:
        if not os.path.exists(p):
            log(f"  WARNING: Missing {p}, skipping adult sub-circuits.", level="WARN")
            return {}

    log("  Loading adult Drosophila feather files (16.8M connections)...")
    conn_df = pd.read_feather(connections_path)
    pre_df = pd.read_feather(pre_counts_path)
    post_df = pd.read_feather(post_counts_path)

    log(f"  Loaded {len(conn_df)} connections, "
        f"{len(pre_df)} pre-synaptic entries, {len(post_df)} post-synaptic entries")

    # Neuropil definitions: name -> list of neuropil column prefixes to combine
    SUBCIRCUIT_DEFS = {
        "optic_lobe_medulla": ["ME_R"],
        "mushroom_body": ["MB_CA_R", "MB_ML_R", "MB_PED_R", "MB_VL_R", "MB(R)"],
        "central_complex": ["EB", "FB", "NO", "PB"],
        "antennal_lobe": ["AL_R"],
        "subesophageal_zone": ["SEZ", "SEZ_R"],
        "lateral_horn": ["LH_R"],
    }

    results = {}

    for sc_name, neuropil_prefixes in SUBCIRCUIT_DEFS.items():
        adj_path = os.path.join(SUBCIRCUIT_DIR, f"{sc_name}_adjacency.npz")
        meta_path = os.path.join(SUBCIRCUIT_DIR, f"{sc_name}_metadata.json")

        if os.path.exists(adj_path):
            log(f"  {sc_name}: already exists, skipping.")
            results[sc_name] = sp.load_npz(adj_path)
            continue

        log(f"  {sc_name}: extracting neurons for neuropils {neuropil_prefixes}...")

        # Find matching neuropil columns in pre and post count dataframes
        def _find_matching_cols(df, prefixes):
            """Find columns in df that match any of the given neuropil prefixes."""
            matched = []
            for col in df.columns:
                for pref in prefixes:
                    if col == pref or col.startswith(pref + "_") or col.startswith(pref):
                        # Exclude the neuron ID column
                        if col not in ("root_id", "pt_root_id", "pre_pt_root_id",
                                       "post_pt_root_id", "cell_id"):
                            matched.append(col)
                            break
            return list(set(matched))

        # Identify the neuron ID column in pre/post count files
        pre_id_col = None
        for candidate in ["root_id", "pt_root_id", "pre_pt_root_id", "cell_id"]:
            if candidate in pre_df.columns:
                pre_id_col = candidate
                break
        if pre_id_col is None:
            # Use the first column as ID
            pre_id_col = pre_df.columns[0]

        post_id_col = None
        for candidate in ["root_id", "pt_root_id", "post_pt_root_id", "cell_id"]:
            if candidate in post_df.columns:
                post_id_col = candidate
                break
        if post_id_col is None:
            post_id_col = post_df.columns[0]

        # Get neuropil count columns
        pre_neuropil_cols = _find_matching_cols(pre_df, neuropil_prefixes)
        post_neuropil_cols = _find_matching_cols(post_df, neuropil_prefixes)

        log(f"    Found {len(pre_neuropil_cols)} pre-synaptic neuropil cols, "
            f"{len(post_neuropil_cols)} post-synaptic neuropil cols")

        if len(pre_neuropil_cols) == 0 and len(post_neuropil_cols) == 0:
            log(f"    WARNING: No neuropil columns found for {sc_name}, "
                f"trying exact match on 'neuropil' column in connections.", level="WARN")
            # Fallback: filter connections by neuropil column directly
            if "neuropil" in conn_df.columns:
                mask = conn_df["neuropil"].isin(neuropil_prefixes)
                sub_conn = conn_df[mask]
                if len(sub_conn) == 0:
                    # Try prefix matching
                    neuropil_mask = pd.Series(False, index=conn_df.index)
                    for pref in neuropil_prefixes:
                        neuropil_mask |= conn_df["neuropil"].str.startswith(pref, na=False)
                    sub_conn = conn_df[neuropil_mask]

                if len(sub_conn) > 0:
                    neurons = sorted(set(sub_conn["pre_pt_root_id"].unique()) |
                                     set(sub_conn["post_pt_root_id"].unique()))
                    log(f"    Fallback: {len(neurons)} neurons from {len(sub_conn)} connections")
                else:
                    log(f"    SKIPPING {sc_name}: no matching connections found.", level="WARN")
                    continue
            else:
                log(f"    SKIPPING {sc_name}: cannot identify neurons.", level="WARN")
                continue
        else:
            # For each neuron, compute total synapses and neuropil-specific synapses
            # A neuron "belongs" to this neuropil if its majority of pre OR post synapses
            # are in the target neuropil columns.
            pre_neurons = set()
            if pre_neuropil_cols:
                # Sum across all matching neuropil columns for this sub-circuit
                numeric_cols = [c for c in pre_df.columns if c != pre_id_col]
                pre_df_copy = pre_df.copy()
                pre_df_copy["_target_sum"] = pre_df_copy[pre_neuropil_cols].sum(axis=1)
                pre_df_copy["_total_sum"] = pre_df_copy[numeric_cols].select_dtypes(
                    include=[np.number]).sum(axis=1)
                # Neurons with significant presence: at least 20% of their synapses in this region
                # OR they have a substantial absolute count (>10)
                mask = ((pre_df_copy["_target_sum"] / pre_df_copy["_total_sum"].clip(lower=1)) > 0.2) | \
                       (pre_df_copy["_target_sum"] > 10)
                pre_neurons = set(pre_df_copy.loc[mask, pre_id_col].values)

            post_neurons = set()
            if post_neuropil_cols:
                numeric_cols = [c for c in post_df.columns if c != post_id_col]
                post_df_copy = post_df.copy()
                post_df_copy["_target_sum"] = post_df_copy[post_neuropil_cols].sum(axis=1)
                post_df_copy["_total_sum"] = post_df_copy[numeric_cols].select_dtypes(
                    include=[np.number]).sum(axis=1)
                mask = ((post_df_copy["_target_sum"] / post_df_copy["_total_sum"].clip(lower=1)) > 0.2) | \
                       (post_df_copy["_target_sum"] > 10)
                post_neurons = set(post_df_copy.loc[mask, post_id_col].values)

            neurons = sorted(pre_neurons | post_neurons)
            log(f"    Found {len(neurons)} neurons ({len(pre_neurons)} pre, {len(post_neurons)} post)")

            if len(neurons) == 0:
                log(f"    SKIPPING {sc_name}: no neurons found.", level="WARN")
                continue

            # Extract subgraph from full connection table
            neuron_set = set(neurons)
            sub_conn = conn_df[
                conn_df["pre_pt_root_id"].isin(neuron_set) &
                conn_df["post_pt_root_id"].isin(neuron_set)
            ]

        # If too many neurons, subsample to ~5000
        MAX_NEURONS = 5000
        if len(neurons) > MAX_NEURONS:
            log(f"    Subsampling from {len(neurons)} to {MAX_NEURONS} neurons...")
            # Keep neurons with highest total synapse count
            neuron_conn_counts = pd.concat([
                sub_conn.groupby("pre_pt_root_id")["syn_count"].sum().rename("total"),
                sub_conn.groupby("post_pt_root_id")["syn_count"].sum().rename("total"),
            ]).groupby(level=0).sum().sort_values(ascending=False)
            top_neurons = set(neuron_conn_counts.head(MAX_NEURONS).index)
            sub_conn = sub_conn[
                sub_conn["pre_pt_root_id"].isin(top_neurons) &
                sub_conn["post_pt_root_id"].isin(top_neurons)
            ]
            neurons = sorted(top_neurons & (set(sub_conn["pre_pt_root_id"]) |
                                            set(sub_conn["post_pt_root_id"])))

        if len(sub_conn) == 0:
            log(f"    SKIPPING {sc_name}: no connections among selected neurons.", level="WARN")
            continue

        # Aggregate connections: sum syn_count for each (pre, post) pair
        agg = sub_conn.groupby(["pre_pt_root_id", "post_pt_root_id"])["syn_count"].sum().reset_index()

        # Build sparse matrix
        neuron_list = sorted(set(agg["pre_pt_root_id"]) | set(agg["post_pt_root_id"]))
        neuron_to_id = {n: i for i, n in enumerate(neuron_list)}
        N = len(neuron_list)

        rows = np.array([neuron_to_id[x] for x in agg["pre_pt_root_id"]])
        cols = np.array([neuron_to_id[x] for x in agg["post_pt_root_id"]])
        weights = agg["syn_count"].values.astype(np.float64)

        adj = sp.csr_matrix((weights, (rows, cols)), shape=(N, N))

        sp.save_npz(adj_path, adj)
        metadata = {
            "n_neurons": N,
            "n_edges": int(adj.nnz),
            "neuropil_regions": neuropil_prefixes,
            "node_ids": [str(x) for x in neuron_list],
            "source_file": "proofread_connections_783.feather",
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        log(f"  {sc_name}: saved {N} neurons, {adj.nnz} edges")
        results[sc_name] = adj

    return results


def run_phase1(small=False):
    """Phase 1: Standardize all raw connectomes into sparse adjacency matrices."""
    log_phase(1, "STANDARDIZE ALL CONNECTOMES")

    connectomes = {}

    # --- C. elegans hermaphrodite ---
    csv_path = os.path.join(RAW_DIR, "celegans_herm", "chemical_synapses.csv")
    if os.path.exists(csv_path):
        connectomes["celegans_herm"] = standardize_edge_list(csv_path, "celegans_herm")
    else:
        log(f"  WARNING: {csv_path} not found", level="WARN")

    if small:
        log("  --small mode: skipping remaining organisms.")
        return connectomes

    # --- C. elegans male ---
    csv_path = os.path.join(RAW_DIR, "celegans_male", "chemical_synapses.csv")
    if os.path.exists(csv_path):
        connectomes["celegans_male"] = standardize_edge_list(csv_path, "celegans_male")
    else:
        log(f"  WARNING: {csv_path} not found", level="WARN")

    # --- Ciona ---
    csv_path = os.path.join(RAW_DIR, "ciona", "edge_list.csv")
    if os.path.exists(csv_path):
        connectomes["ciona"] = standardize_edge_list(csv_path, "ciona")
    else:
        log(f"  WARNING: {csv_path} not found (may still be in progress)", level="WARN")

    # --- Drosophila larva ---
    larva_csv = os.path.join(
        RAW_DIR, "drosophila_larva", "Supplementary-Data-S1",
        "all-all_connectivity_matrix.csv"
    )
    if os.path.exists(larva_csv):
        connectomes["drosophila_larva"] = standardize_drosophila_larva()
    else:
        log(f"  WARNING: {larva_csv} not found", level="WARN")

    # --- Drosophila adult sub-circuits ---
    adult_subcircuits = standardize_drosophila_adult_subcircuits()
    for sc_name, adj in adult_subcircuits.items():
        connectomes[f"adult_{sc_name}"] = adj

    log(f"\n  Phase 1 complete: {len(connectomes)} connectomes standardized.")
    return connectomes


# ============================================================================
# PHASE 2: GENERATE CONTROLS
# ============================================================================

def load_all_biological_connectomes():
    """Load all biological connectomes from processed/ and subcircuits/ directories."""
    connectomes = {}

    # Whole organisms from processed/
    for fname in sorted(os.listdir(PROCESSED_DIR)):
        if fname.endswith("_adjacency.npz"):
            name = fname.replace("_adjacency.npz", "")
            # Skip controls (they contain control type names)
            if any(ct in name for ct in ["_erdos_renyi", "_barabasi_albert",
                                          "_watts_strogatz", "_degree_preserved"]):
                continue
            connectomes[name] = sp.load_npz(os.path.join(PROCESSED_DIR, fname))

    # Sub-circuits
    for fname in sorted(os.listdir(SUBCIRCUIT_DIR)):
        if fname.endswith("_adjacency.npz"):
            name = fname.replace("_adjacency.npz", "")
            if any(ct in name for ct in ["_erdos_renyi", "_barabasi_albert",
                                          "_watts_strogatz", "_degree_preserved"]):
                continue
            connectomes[f"adult_{name}"] = sp.load_npz(
                os.path.join(SUBCIRCUIT_DIR, fname)
            )

    return connectomes


def run_phase2(small=False):
    """Phase 2: Generate control graphs for each biological connectome."""
    log_phase(2, "GENERATE CONTROLS")

    from models.controls import generate_all_controls

    bio_connectomes = load_all_biological_connectomes()

    if small:
        # Only keep celegans_herm
        bio_connectomes = {k: v for k, v in bio_connectomes.items()
                          if k == "celegans_herm"}

    log(f"  Generating controls for {len(bio_connectomes)} biological connectomes...")

    control_types = ["erdos_renyi", "barabasi_albert", "watts_strogatz", "degree_preserved"]

    for name, bio_adj in bio_connectomes.items():
        N = bio_adj.shape[0]
        log(f"\n  {name} (N={N}):")

        # Determine output directory (processed for whole organisms, subcircuits for sub-circuits)
        if name.startswith("adult_"):
            out_dir = SUBCIRCUIT_DIR
            base_name = name.replace("adult_", "")
        else:
            out_dir = PROCESSED_DIR
            base_name = name

        # Check if all controls already exist
        all_exist = all(
            os.path.exists(os.path.join(out_dir, f"{base_name}_{ct}_adjacency.npz"))
            for ct in control_types
        )
        if all_exist:
            log(f"    All controls exist, skipping.")
            continue

        # For very large networks (>5000), degree_preserved_shuffle can be slow
        if N > 10000:
            log(f"    WARNING: N={N} is large. Controls may take a while.", level="WARN")

        try:
            controls = generate_all_controls(bio_adj, seed=42)
            for ct_name, ct_adj in controls.items():
                ct_path = os.path.join(out_dir, f"{base_name}_{ct_name}_adjacency.npz")
                if not os.path.exists(ct_path):
                    sp.save_npz(ct_path, ct_adj)
                    log(f"    Saved {ct_name} (N={ct_adj.shape[0]}, edges={ct_adj.nnz})")
                else:
                    log(f"    {ct_name} already exists.")
        except Exception as e:
            log(f"    ERROR generating controls for {name}: {e}", level="ERROR")
            traceback.print_exc()

    log("\n  Phase 2 complete.")


# ============================================================================
# PHASE 3: COMPUTE GRAPH METRICS
# ============================================================================

def load_all_adjacency_matrices():
    """Load ALL adjacency matrices (biological + controls) from both directories."""
    all_adj = {}

    for dirpath, dirnames, filenames in os.walk(PROCESSED_DIR):
        for fname in sorted(filenames):
            if fname.endswith("_adjacency.npz"):
                name = fname.replace("_adjacency.npz", "")
                all_adj[name] = sp.load_npz(os.path.join(dirpath, fname))

    for dirpath, dirnames, filenames in os.walk(SUBCIRCUIT_DIR):
        for fname in sorted(filenames):
            if fname.endswith("_adjacency.npz"):
                name = fname.replace("_adjacency.npz", "")
                # Prefix sub-circuit names with 'adult_' for consistency
                full_name = f"adult_{name}"
                all_adj[full_name] = sp.load_npz(os.path.join(dirpath, fname))

    return all_adj


def run_phase3(small=False):
    """Phase 3: Compute graph-theoretic metrics for all adjacency matrices."""
    log_phase(3, "COMPUTE GRAPH METRICS")

    from analysis.graph_metrics import compute_metrics_for_all

    all_adj = load_all_adjacency_matrices()

    if small:
        # Only keep celegans_herm and its controls
        all_adj = {k: v for k, v in all_adj.items() if "celegans_herm" in k}

    log(f"  Computing metrics for {len(all_adj)} networks...")

    output_path = os.path.join(RESULTS_DIR, "graph_metrics.csv")
    metrics_df = compute_metrics_for_all(all_adj, output_path=output_path)

    log(f"\n  Phase 3 complete: {len(metrics_df)} networks analyzed.")
    return metrics_df


# ============================================================================
# PHASE 4: RUN BENCHMARKS
# ============================================================================

# Benchmark registry: (function_module, function_name, task_key, model_type, default_kwargs)
BENCHMARK_REGISTRY = [
    ("benchmarks.mnist", "run_mnist", "MNIST", "feedforward",
     {"epochs": 20, "lr": 1e-3, "batch_size": 128}),
    ("benchmarks.fashion_mnist", "run_fashion_mnist", "FashionMNIST", "feedforward",
     {"epochs": 20, "lr": 1e-3, "batch_size": 128}),
    ("benchmarks.cifar10", "run_cifar10", "CIFAR10", "feedforward",
     {"epochs": 50, "lr": 1e-3, "batch_size": 128}),
    ("benchmarks.sequential_mnist", "run_sequential_mnist", "SequentialMNIST", "sequential",
     {"epochs": 20, "lr": 1e-3, "batch_size": 128}),
    ("benchmarks.audio", "run_audio", "Audio", "feedforward",
     {"epochs": 20, "lr": 1e-3, "batch_size": 64}),
    ("benchmarks.cartpole", "run_cartpole", "CartPole", "rl",
     {"n_episodes": 500, "lr": 1e-3}),
]

SMALL_BENCHMARKS = ["MNIST"]  # Subset for --small mode
SEEDS = [42, 43, 44]
SMALL_SEEDS = [42]
SMALL_EPOCHS = 5
GPU_VRAM_BYTES = 12 * (1024 ** 3)  # 12 GB assumed


def _get_device():
    """Get the best available torch device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def _check_gpu_fit(N, batch_size, device):
    """Check if a network of size N fits in GPU memory.

    The W_rec buffer is N*N*4 bytes (float32). Plus model params, activations, etc.
    Conservative estimate: need ~3x the buffer size for forward + backward.
    """
    if device != "cuda":
        return True, batch_size

    import torch

    buffer_bytes = N * N * 4  # W_rec float32
    # Rough estimate: model needs ~3x buffer + batch activations
    estimated_total = buffer_bytes * 3 + batch_size * N * 4 * 10

    if estimated_total > GPU_VRAM_BYTES * 0.85:
        # Try reducing batch size
        for reduced_bs in [64, 32, 16, 8]:
            reduced_total = buffer_bytes * 3 + reduced_bs * N * 4 * 10
            if reduced_total < GPU_VRAM_BYTES * 0.85:
                return True, reduced_bs
        # Still too large -- skip
        return False, batch_size

    return True, batch_size


def _classify_network(name):
    """Determine organism name and type (biological/control) from network name."""
    control_suffixes = ["_erdos_renyi", "_barabasi_albert",
                        "_watts_strogatz", "_degree_preserved"]
    net_type = "biological"
    organism = name
    for suffix in control_suffixes:
        if name.endswith(suffix):
            net_type = suffix.lstrip("_")
            organism = name[: -len(suffix)]
            break
    return organism, net_type


def _import_benchmark(module_path, func_name):
    """Dynamically import a benchmark function."""
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def run_phase4(small=False):
    """Phase 4: Run all benchmarks on all adjacency matrices."""
    log_phase(4, "RUN BENCHMARKS")

    import torch

    device = _get_device()
    log(f"  Device: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        log(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Load all adjacency matrices
    all_adj = load_all_adjacency_matrices()

    if small:
        all_adj = {k: v for k, v in all_adj.items() if "celegans_herm" in k}

    seeds = SMALL_SEEDS if small else SEEDS
    benchmarks = [b for b in BENCHMARK_REGISTRY
                  if small and b[2] in SMALL_BENCHMARKS or not small]

    # Count total runs
    total_runs = len(all_adj) * len(benchmarks) * len(seeds)
    log(f"  Total planned runs: {len(all_adj)} networks x {len(benchmarks)} tasks x "
        f"{len(seeds)} seeds = {total_runs}")

    # Load existing results to support resumption
    results_path = os.path.join(RESULTS_DIR, "all_results.csv")
    existing_results = []
    completed_keys = set()
    if os.path.exists(results_path):
        existing_df = pd.read_csv(results_path)
        existing_results = existing_df.to_dict("records")
        for row in existing_results:
            key = (str(row.get("name", "")), str(row.get("task", "")),
                   str(row.get("seed", "")))
            completed_keys.add(key)
        log(f"  Resuming: {len(completed_keys)} runs already completed.")

    new_results = []
    run_count = 0
    skip_count = 0
    error_count = 0

    for net_name, adj in sorted(all_adj.items()):
        N = adj.shape[0]
        organism, net_type = _classify_network(net_name)

        for bm_module, bm_func_name, task_key, model_type, default_kwargs in benchmarks:
            for seed in seeds:
                run_count += 1
                run_key = (net_name, task_key, str(seed))

                # Skip if already done
                if run_key in completed_keys:
                    skip_count += 1
                    continue

                # Check GPU memory
                batch_size = default_kwargs.get("batch_size", 128)
                fits, adjusted_bs = _check_gpu_fit(N, batch_size, device)
                if not fits:
                    log(f"  [{run_count}/{total_runs}] SKIP {net_name} x {task_key} "
                        f"(N={N} too large for GPU)", level="WARN")
                    skip_count += 1
                    continue

                log(f"  [{run_count}/{total_runs}] {net_name} x {task_key} "
                    f"(seed={seed}, N={N}, bs={adjusted_bs})")

                try:
                    bm_func = _import_benchmark(bm_module, bm_func_name)

                    kwargs = dict(default_kwargs)
                    kwargs["name"] = net_name
                    kwargs["device"] = device
                    kwargs["seed"] = seed

                    if "batch_size" in kwargs:
                        kwargs["batch_size"] = adjusted_bs

                    if small and "epochs" in kwargs:
                        kwargs["epochs"] = SMALL_EPOCHS
                    if small and "n_episodes" in kwargs:
                        kwargs["n_episodes"] = 50

                    result = bm_func(adj, **kwargs)

                    # Add classification columns
                    result["organism"] = organism
                    result["type"] = net_type

                    new_results.append(result)
                    completed_keys.add(run_key)

                    acc = result.get("accuracy", "N/A")
                    t = result.get("train_time_sec", "N/A")
                    log(f"    -> accuracy={acc}, time={t}s")

                except Exception as e:
                    error_count += 1
                    log(f"    ERROR: {e}", level="ERROR")
                    traceback.print_exc()

                # GPU memory cleanup
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

                # Save intermediate results every 10 new runs
                if len(new_results) > 0 and len(new_results) % 10 == 0:
                    _save_results(existing_results + new_results, results_path)
                    log(f"    [checkpoint] Saved {len(existing_results) + len(new_results)} results")

    # Final save
    all_results = existing_results + new_results
    _save_results(all_results, results_path)

    log(f"\n  Phase 4 complete:")
    log(f"    Total runs:    {run_count}")
    log(f"    Completed:     {len(new_results)} new + {skip_count - (skip_count - len([k for k in completed_keys]))} resumed")
    log(f"    Skipped:       {skip_count}")
    log(f"    Errors:        {error_count}")
    log(f"    Results saved: {results_path}")


def _save_results(results_list, path):
    """Save results list to CSV, handling heterogeneous keys."""
    if not results_list:
        return
    df = pd.DataFrame(results_list)
    # Ensure consistent column ordering
    priority_cols = ["name", "organism", "type", "task", "n_neurons", "accuracy",
                     "final_loss", "train_time_sec", "learnable_params", "fixed_params",
                     "epochs", "lr", "seed"]
    existing_priority = [c for c in priority_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[existing_priority + other_cols]
    df.to_csv(path, index=False)


# ============================================================================
# PHASE 5: ANALYSIS
# ============================================================================

def run_phase5(small=False):
    """Phase 5: Run all analysis modules on collected results."""
    log_phase(5, "ANALYSIS")

    results_csv = os.path.join(RESULTS_DIR, "all_results.csv")
    metrics_csv = os.path.join(RESULTS_DIR, "graph_metrics.csv")

    if not os.path.exists(results_csv):
        log("  ERROR: all_results.csv not found. Run Phase 4 first.", level="ERROR")
        return

    # --- 5a: Scaling Laws ---
    log("\n  --- 5a: Scaling Law Analysis ---")
    try:
        from analysis.scaling_law import fit_all_tasks
        scaling_df = fit_all_tasks(results_csv)
        log(f"  Scaling law fits saved.")
    except Exception as e:
        log(f"  ERROR in scaling law analysis: {e}", level="ERROR")
        traceback.print_exc()

    # --- 5b: Graph Metric Correlations ---
    log("\n  --- 5b: Graph Metric Correlations ---")
    if os.path.exists(metrics_csv):
        try:
            from analysis.correlation import compute_all_task_correlations
            corr_df = compute_all_task_correlations(metrics_csv, results_csv)
            log(f"  Correlation analysis saved.")
        except Exception as e:
            log(f"  ERROR in correlation analysis: {e}", level="ERROR")
            traceback.print_exc()
    else:
        log("  WARNING: graph_metrics.csv not found. Run Phase 3 first.", level="WARN")

    # --- 5c: Sub-circuit Specialization ---
    log("\n  --- 5c: Sub-circuit Specialization ---")
    try:
        from analysis.specialization import test_specialization
        spec_results = test_specialization(results_csv)
        if spec_results:
            spec_path = os.path.join(RESULTS_DIR, "specialization_results.json")
            with open(spec_path, "w") as f:
                json.dump(spec_results, f, indent=2, default=str)
            log(f"  Specialization results saved to {spec_path}")
    except Exception as e:
        log(f"  ERROR in specialization analysis: {e}", level="ERROR")
        traceback.print_exc()

    # --- 5d: Sexual Dimorphism ---
    log("\n  --- 5d: Sexual Dimorphism (C. elegans) ---")
    try:
        from analysis.specialization import test_sexual_dimorphism
        dim_results = test_sexual_dimorphism(results_csv)
        if dim_results:
            dim_path = os.path.join(RESULTS_DIR, "dimorphism_results.json")
            with open(dim_path, "w") as f:
                json.dump(dim_results, f, indent=2, default=str)
            log(f"  Dimorphism results saved to {dim_path}")
    except Exception as e:
        log(f"  ERROR in dimorphism analysis: {e}", level="ERROR")
        traceback.print_exc()

    log("\n  Phase 5 complete.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-species BPU experiment runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase", nargs="+", default=["all"],
        help="Phases to run: 1, 2, 3, 4, 5, or 'all' (default: all)"
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Quick test mode: only C. elegans herm, only MNIST, 1 seed, 5 epochs"
    )
    args = parser.parse_args()

    # Determine which phases to run
    if "all" in args.phase:
        phases = [1, 2, 3, 4, 5]
    else:
        phases = sorted(set(int(p) for p in args.phase))

    # Print banner
    print("\n" + "=" * 72)
    print("  CROSS-SPECIES BPU EXPERIMENT RUNNER")
    print("=" * 72)
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Phases to run: {phases}")
    log(f"Small mode: {args.small}")
    log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    t_start = time.time()

    # --- Run selected phases ---
    if 1 in phases:
        try:
            run_phase1(small=args.small)
        except Exception as e:
            log(f"PHASE 1 FAILED: {e}", level="ERROR")
            traceback.print_exc()

    if 2 in phases:
        try:
            run_phase2(small=args.small)
        except Exception as e:
            log(f"PHASE 2 FAILED: {e}", level="ERROR")
            traceback.print_exc()

    if 3 in phases:
        try:
            run_phase3(small=args.small)
        except Exception as e:
            log(f"PHASE 3 FAILED: {e}", level="ERROR")
            traceback.print_exc()

    if 4 in phases:
        try:
            run_phase4(small=args.small)
        except Exception as e:
            log(f"PHASE 4 FAILED: {e}", level="ERROR")
            traceback.print_exc()

    if 5 in phases:
        try:
            run_phase5(small=args.small)
        except Exception as e:
            log(f"PHASE 5 FAILED: {e}", level="ERROR")
            traceback.print_exc()

    # --- Final summary ---
    elapsed = time.time() - t_start
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    print("\n" + "=" * 72)
    log(f"ALL PHASES COMPLETE")
    log(f"Total wall time: {elapsed_str}")
    log(f"Results directory: {RESULTS_DIR}")

    # List output files
    if os.path.exists(RESULTS_DIR):
        output_files = sorted(os.listdir(RESULTS_DIR))
        if output_files:
            log("Output files:")
            for f in output_files:
                fpath = os.path.join(RESULTS_DIR, f)
                size_kb = os.path.getsize(fpath) / 1024
                log(f"  {f} ({size_kb:.1f} KB)")

    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
