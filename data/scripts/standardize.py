"""standardize.py — Convert all raw connectomes to standardized sparse matrices.

Converts each connectome to:
  - {name}_adjacency.npz: scipy CSR sparse matrix, weights normalized to [0,1]
  - {name}_metadata.json: neuron count, synapse count, density, neuron IDs

Usage:
  python data/scripts/standardize.py
"""
import hashlib
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import pandas as pd
import os
import json
import glob
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "processed")
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw")


def standardize_connectome(
    name,
    edge_df,
    pre_col,
    post_col,
    weight_col=None,
    output_dir=None,
    provenance=None,
):
    """Convert edge list DataFrame to standardized sparse adjacency matrix.

    Args:
        name: organism identifier (e.g., 'celegans_herm')
        edge_df: DataFrame with edge list
        pre_col: column name for presynaptic neuron ID
        post_col: column name for postsynaptic neuron ID
        weight_col: column name for synapse count/weight (None = all edges weight 1)
        output_dir: where to save (default: data/processed/)
        provenance: measured-source record containing source_kind, source_url,
            and citation. The standardized artifact digest is added here.

    Returns:
        tuple: (adj_sparse, metadata_dict)

    Saves:
        {output_dir}/{name}_adjacency.npz
        {output_dir}/{name}_metadata.json
    """
    if output_dir is None:
        output_dir = PROCESSED_DIR
    os.makedirs(output_dir, exist_ok=True)
    if provenance is None:
        raise ValueError(f"{name} is missing provenance metadata")
    for field in ("source_kind", "source_url", "citation"):
        if not provenance.get(field):
            raise ValueError(f"{name} provenance is missing {field}")

    # Clean edge dataframe
    edge_df = edge_df.dropna(subset=[pre_col, post_col])

    # Build neuron ID mapping (sorted for reproducibility)
    all_neurons = sorted(set(edge_df[pre_col].astype(str).unique()) |
                         set(edge_df[post_col].astype(str).unique()))
    neuron_to_idx = {n: i for i, n in enumerate(all_neurons)}
    N = len(all_neurons)

    # Map neuron IDs to indices
    rows = edge_df[pre_col].astype(str).map(neuron_to_idx).values
    cols = edge_df[post_col].astype(str).map(neuron_to_idx).values

    # Get weights
    if weight_col and weight_col in edge_df.columns:
        weights = edge_df[weight_col].values.astype(np.float32)
    else:
        weights = np.ones(len(edge_df), dtype=np.float32)

    # Handle duplicate edges (aggregate by summing weights)
    # Use pandas for clean aggregation
    agg_df = pd.DataFrame({'row': rows, 'col': cols, 'weight': weights})
    agg_df = agg_df.groupby(['row', 'col'])['weight'].sum().reset_index()

    # Build sparse matrix
    adj = csr_matrix(
        (agg_df['weight'].values, (agg_df['row'].values.astype(int), agg_df['col'].values.astype(int))),
        shape=(N, N)
    )

    # Normalize weights to [0, 1]
    max_weight = adj.max()
    if max_weight > 0:
        adj = adj / max_weight

    # Save sparse matrix
    artifact_path = os.path.join(output_dir, f"{name}_adjacency.npz")
    save_npz(artifact_path, adj)
    with open(artifact_path, "rb") as artifact:
        artifact_sha256 = hashlib.sha256(artifact.read()).hexdigest()

    # Save metadata
    metadata = {
        "name": name,
        "n_neurons": N,
        "n_synapses": int(adj.nnz),
        "n_edges_raw": len(edge_df),
        "density": float(adj.nnz / (N * N)) if N > 0 else 0,
        "max_weight_raw": float(max_weight) if max_weight > 0 else 0,
        "neuron_ids": all_neurons[:20],  # First 20 for reference
        "total_neuron_ids": len(all_neurons),
        "columns_used": {
            "pre": pre_col,
            "post": post_col,
            "weight": weight_col
        },
        "provenance": {
            **provenance,
            "artifact_sha256": artifact_sha256,
        },
    }
    with open(os.path.join(output_dir, f"{name}_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"  [{name}] {N} neurons, {adj.nnz} synapses, density={metadata['density']:.6f}")
    return adj, metadata


def auto_detect_columns(df):
    """Try to auto-detect pre/post/weight columns from a DataFrame."""
    # Common pre-synaptic column names
    pre_candidates = ['pre', 'pre_id', 'pre_root_id', 'source', 'from', 'presynaptic',
                      'pre_pt_root_id', 'pre_neuron', 'neuron_1', 'from_id']
    # Common post-synaptic column names
    post_candidates = ['post', 'post_id', 'post_root_id', 'target', 'to', 'postsynaptic',
                       'post_pt_root_id', 'post_neuron', 'neuron_2', 'to_id']
    # Common weight column names
    weight_candidates = ['weight', 'syn_count', 'synapses', 'n_synapses', 'count',
                         'strength', 'n_connections', 'num_synapses']

    pre_col = None
    post_col = None
    weight_col = None

    for i, col in enumerate(df.columns):
        cl = col.lower().strip()
        if cl in pre_candidates and pre_col is None:
            pre_col = col
        elif cl in post_candidates and post_col is None:
            post_col = col
        elif cl in weight_candidates and weight_col is None:
            weight_col = col

    return pre_col, post_col, weight_col


def standardize_all():
    """Standardize all raw connectomes found in data/raw/."""
    print("=" * 60)
    print("CONNECTOME STANDARDIZATION PIPELINE")
    print("=" * 60)

    organisms = {}

    # Scan for CSV files in each organism directory
    for organism_dir in sorted(glob.glob(os.path.join(RAW_DIR, "*"))):
        if not os.path.isdir(organism_dir):
            continue
        name = os.path.basename(organism_dir)

        # Skip non-organism directories
        if name in ['FlyConnectome', 'scripts']:
            continue

        provenance_path = os.path.join(organism_dir, "provenance.json")
        if not os.path.exists(provenance_path):
            print(f"\n[{name}] Missing provenance.json - skipping")
            continue
        with open(provenance_path, encoding="utf-8") as handle:
            provenance = json.load(handle)

        csv_files = glob.glob(os.path.join(organism_dir, "*.csv"))
        parquet_files = glob.glob(os.path.join(organism_dir, "*.parquet"))

        if not csv_files and not parquet_files:
            print(f"\n[{name}] No CSV/Parquet files found in {organism_dir} — skipping")
            continue

        print(f"\n[{name}] Found files:")
        for f in csv_files + parquet_files:
            print(f"  - {os.path.basename(f)}")

        # Try to load and standardize
        for fpath in csv_files:
            fname = os.path.basename(fpath).lower()
            if 'synapse' in fname or 'chemical' in fname or 'edge' in fname or 'connect' in fname:
                try:
                    df = pd.read_csv(fpath)
                    print(f"  Loading {os.path.basename(fpath)}: {len(df)} rows, columns: {list(df.columns)}")

                    pre_col, post_col, weight_col = auto_detect_columns(df)

                    if pre_col and post_col:
                        print(f"  Auto-detected: pre={pre_col}, post={post_col}, weight={weight_col}")
                        adj, meta = standardize_connectome(
                            name,
                            df,
                            pre_col,
                            post_col,
                            weight_col,
                            provenance=provenance,
                        )
                        organisms[name] = (adj, meta)
                    else:
                        print(f"  WARNING: Could not auto-detect columns. Available: {list(df.columns)}")
                        print(f"  Please add manual column mapping for {name}")
                except Exception as e:
                    print(f"  ERROR processing {fpath}: {e}")

        for fpath in parquet_files:
            fname = os.path.basename(fpath).lower()
            try:
                df = pd.read_parquet(fpath)
                print(f"  Loading {os.path.basename(fpath)}: {len(df)} rows, columns: {list(df.columns)}")

                pre_col, post_col, weight_col = auto_detect_columns(df)

                if pre_col and post_col:
                    print(f"  Auto-detected: pre={pre_col}, post={post_col}, weight={weight_col}")
                    adj, meta = standardize_connectome(
                        name,
                        df,
                        pre_col,
                        post_col,
                        weight_col,
                        provenance=provenance,
                    )
                    organisms[name] = (adj, meta)
                else:
                    print(f"  WARNING: Could not auto-detect columns for {fpath}")
            except Exception as e:
                print(f"  ERROR processing {fpath}: {e}")

    print(f"\n{'=' * 60}")
    print(f"STANDARDIZATION COMPLETE: {len(organisms)} organisms processed")
    print(f"{'=' * 60}")

    for name, (adj, meta) in organisms.items():
        print(f"  {name}: {meta['n_neurons']} neurons, {meta['n_synapses']} synapses")

    return organisms


def verify_processed():
    """Verify all processed connectomes load correctly."""
    print("\nVERIFYING PROCESSED CONNECTOMES:")
    for f in sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_adjacency.npz"))):
        name = os.path.basename(f).replace("_adjacency.npz", "")
        adj = load_npz(f)
        meta_path = f.replace("_adjacency.npz", "_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as mf:
                meta = json.load(mf)
            print(f"  {name}: shape={adj.shape}, nnz={adj.nnz}, density={meta['density']:.6f}")
        else:
            print(f"  {name}: shape={adj.shape}, nnz={adj.nnz} (no metadata)")


if __name__ == "__main__":
    standardize_all()
    verify_processed()
