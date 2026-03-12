"""graph_metrics.py — Compute comprehensive graph-theoretic properties for connectomes.

Computes: degree distributions, clustering, path length, small-world index,
modularity, reciprocity, betweenness centrality, spectral gap, rich-club, density.

Usage:
    python analysis/graph_metrics.py
"""
import networkx as nx
import numpy as np
from scipy.sparse import issparse, load_npz
from scipy.sparse.linalg import eigsh
import pandas as pd
import json
import os
import glob
import sys
import warnings
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    warnings.warn("python-louvain not installed. Modularity will be skipped.")


def compute_all_metrics(adj_sparse, name="unnamed", verbose=True):
    """Compute full suite of graph-theoretic metrics for a connectome.

    Args:
        adj_sparse: scipy sparse matrix or numpy array (N x N adjacency)
        name: identifier string
        verbose: print progress

    Returns:
        dict of metric_name -> value
    """
    if issparse(adj_sparse):
        G = nx.from_scipy_sparse_array(adj_sparse, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_array(np.array(adj_sparse), create_using=nx.DiGraph)

    N = G.number_of_nodes()
    M = G.number_of_edges()

    if verbose:
        print(f"  [{name}] Computing metrics for {N} nodes, {M} edges...")

    metrics = {"name": name, "n_neurons": N, "n_edges": M}

    # === 1. DEGREE DISTRIBUTIONS ===
    in_degrees = np.array([d for _, d in G.in_degree()])
    out_degrees = np.array([d for _, d in G.out_degree()])
    total_degrees = in_degrees + out_degrees

    metrics["mean_in_degree"] = float(np.mean(in_degrees))
    metrics["std_in_degree"] = float(np.std(in_degrees))
    metrics["max_in_degree"] = int(np.max(in_degrees))
    metrics["mean_out_degree"] = float(np.mean(out_degrees))
    metrics["std_out_degree"] = float(np.std(out_degrees))
    metrics["max_out_degree"] = int(np.max(out_degrees))
    metrics["mean_total_degree"] = float(np.mean(total_degrees))

    # Degree distribution skewness (indicator of hub structure)
    if np.std(total_degrees) > 0:
        from scipy.stats import skew, kurtosis
        metrics["degree_skewness"] = float(skew(total_degrees))
        metrics["degree_kurtosis"] = float(kurtosis(total_degrees))
    else:
        metrics["degree_skewness"] = 0.0
        metrics["degree_kurtosis"] = 0.0

    # === 2. CLUSTERING COEFFICIENT ===
    G_undir = G.to_undirected()
    metrics["clustering_coeff"] = float(nx.average_clustering(G_undir))

    # Directed clustering (transitivity)
    try:
        metrics["transitivity"] = float(nx.transitivity(G))
    except Exception:
        metrics["transitivity"] = None

    # === 3. CHARACTERISTIC PATH LENGTH ===
    try:
        if nx.is_connected(G_undir):
            metrics["char_path_length"] = float(nx.average_shortest_path_length(G_undir))
            metrics["largest_cc_fraction"] = 1.0
        else:
            largest_cc = max(nx.connected_components(G_undir), key=len)
            subG = G_undir.subgraph(largest_cc)
            metrics["char_path_length"] = float(nx.average_shortest_path_length(subG))
            metrics["largest_cc_fraction"] = float(len(largest_cc) / N)
    except Exception as e:
        metrics["char_path_length"] = None
        metrics["largest_cc_fraction"] = None
        if verbose:
            print(f"    Warning: path length computation failed: {e}")

    # === 4. SMALL-WORLD INDEX ===
    try:
        density = M / (N * (N - 1)) if N > 1 else 0
        G_rand = nx.erdos_renyi_graph(N, density, seed=42)
        C_rand = nx.average_clustering(G_rand)
        if C_rand == 0:
            C_rand = 1e-10

        if nx.is_connected(G_rand):
            L_rand = nx.average_shortest_path_length(G_rand)
        else:
            largest_cc_rand = max(nx.connected_components(G_rand), key=len)
            L_rand = nx.average_shortest_path_length(G_rand.subgraph(largest_cc_rand))

        C_bio = metrics["clustering_coeff"]
        L_bio = metrics["char_path_length"]

        if C_rand > 0 and L_bio and L_bio > 0:
            metrics["small_world_sigma"] = float((C_bio / C_rand) * (L_rand / L_bio))
        else:
            metrics["small_world_sigma"] = None
    except Exception as e:
        metrics["small_world_sigma"] = None
        if verbose:
            print(f"    Warning: small-world computation failed: {e}")

    # === 5. MODULARITY (Louvain) ===
    if HAS_LOUVAIN:
        try:
            partition = community_louvain.best_partition(G_undir, random_state=42)
            metrics["modularity"] = float(community_louvain.modularity(partition, G_undir))
            metrics["n_communities"] = len(set(partition.values()))
        except Exception as e:
            metrics["modularity"] = None
            metrics["n_communities"] = None
            if verbose:
                print(f"    Warning: modularity computation failed: {e}")
    else:
        metrics["modularity"] = None
        metrics["n_communities"] = None

    # === 6. RECIPROCITY ===
    try:
        metrics["reciprocity"] = float(nx.reciprocity(G))
    except Exception:
        metrics["reciprocity"] = None

    # === 7. BETWEENNESS CENTRALITY ===
    try:
        # For large graphs, use approximate betweenness
        if N > 5000:
            bc = nx.betweenness_centrality(G, k=min(500, N))
        else:
            bc = nx.betweenness_centrality(G)
        bc_values = list(bc.values())
        metrics["mean_betweenness"] = float(np.mean(bc_values))
        metrics["max_betweenness"] = float(np.max(bc_values))
        metrics["std_betweenness"] = float(np.std(bc_values))
    except Exception as e:
        metrics["mean_betweenness"] = None
        metrics["max_betweenness"] = None
        metrics["std_betweenness"] = None
        if verbose:
            print(f"    Warning: betweenness computation failed: {e}")

    # === 8. SPECTRAL GAP ===
    try:
        if issparse(adj_sparse):
            k = min(6, N - 2)
            if k >= 2:
                eigenvalues = eigsh(adj_sparse.astype(float), k=k, return_eigenvectors=False)
                sorted_eigs = sorted(np.abs(eigenvalues), reverse=True)
                metrics["spectral_gap"] = float(sorted_eigs[0] - sorted_eigs[1])
                metrics["spectral_radius"] = float(sorted_eigs[0])
            else:
                metrics["spectral_gap"] = None
                metrics["spectral_radius"] = None
        else:
            eigenvalues = np.linalg.eigvalsh(adj_sparse.astype(float))
            sorted_eigs = sorted(np.abs(eigenvalues), reverse=True)
            metrics["spectral_gap"] = float(sorted_eigs[0] - sorted_eigs[1])
            metrics["spectral_radius"] = float(sorted_eigs[0])
    except Exception as e:
        metrics["spectral_gap"] = None
        metrics["spectral_radius"] = None
        if verbose:
            print(f"    Warning: spectral computation failed: {e}")

    # === 9. RICH-CLUB COEFFICIENT ===
    try:
        rc = nx.rich_club_coefficient(G_undir, normalized=False)
        k_values = sorted(rc.keys())
        if k_values:
            k_25 = k_values[len(k_values) // 4]
            k_50 = k_values[len(k_values) // 2]
            k_75 = k_values[3 * len(k_values) // 4]
            metrics["rich_club_k25"] = float(rc[k_25])
            metrics["rich_club_k50"] = float(rc[k_50])
            metrics["rich_club_k75"] = float(rc[k_75])
        else:
            metrics["rich_club_k25"] = None
            metrics["rich_club_k50"] = None
            metrics["rich_club_k75"] = None
    except Exception as e:
        metrics["rich_club_k25"] = None
        metrics["rich_club_k50"] = None
        metrics["rich_club_k75"] = None
        if verbose:
            print(f"    Warning: rich-club computation failed: {e}")

    # === 10. DENSITY ===
    metrics["density"] = float(nx.density(G))

    # === 11. ASSORTATIVITY ===
    try:
        metrics["degree_assortativity"] = float(nx.degree_assortativity_coefficient(G))
    except Exception:
        metrics["degree_assortativity"] = None

    # === 12. GLOBAL EFFICIENCY ===
    try:
        if N <= 5000:
            metrics["global_efficiency"] = float(nx.global_efficiency(G_undir))
        else:
            metrics["global_efficiency"] = None  # Too slow for large graphs
    except Exception:
        metrics["global_efficiency"] = None

    if verbose:
        C = metrics.get("clustering_coeff", "N/A")
        M_val = metrics.get("modularity", "N/A")
        SW = metrics.get("small_world_sigma", "N/A")
        print(f"    Done: clustering={C:.4f}, modularity={M_val}, SW={SW}")

    return metrics


def compute_metrics_for_all(connectomes_dict, output_path=None):
    """Compute metrics for all connectomes and save to CSV.

    Args:
        connectomes_dict: {name: scipy_sparse_matrix}
        output_path: where to save CSV (default: results/graph_metrics.csv)

    Returns:
        DataFrame with all metrics
    """
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "graph_metrics.csv"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_metrics = []
    for i, (name, adj) in enumerate(connectomes_dict.items()):
        print(f"\n[{i+1}/{len(connectomes_dict)}] Processing {name}...")
        m = compute_all_metrics(adj, name)
        all_metrics.append(m)

    df = pd.DataFrame(all_metrics)
    df.to_csv(output_path, index=False)
    print(f"\nSaved metrics for {len(all_metrics)} networks to {output_path}")
    return df


def load_all_processed_connectomes():
    """Load all standardized connectomes from data/processed/ and data/subcircuits/."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    connectomes = {}

    for pattern in ["data/processed/*_adjacency.npz", "data/subcircuits/*_adjacency.npz"]:
        for f in sorted(glob.glob(os.path.join(base, pattern))):
            name = os.path.basename(f).replace("_adjacency.npz", "")
            if "subcircuits" in f:
                name = "adult_" + name
            connectomes[name] = load_npz(f)

    return connectomes


if __name__ == "__main__":
    print("Loading processed connectomes...")
    connectomes = load_all_processed_connectomes()

    if not connectomes:
        print("No processed connectomes found. Run standardize.py first.")
        sys.exit(1)

    print(f"Found {len(connectomes)} connectomes")
    compute_metrics_for_all(connectomes)
