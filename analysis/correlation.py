"""correlation.py — Graph metrics <-> BPU performance correlations.

Tests H3: Which graph-theoretic properties predict computational capability?
Computes Spearman rank correlations with Bonferroni correction.

Usage:
    python analysis/correlation.py
"""
import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# Graph metrics to correlate with BPU performance
METRIC_COLS = [
    "clustering_coeff",
    "char_path_length",
    "small_world_sigma",
    "modularity",
    "reciprocity",
    "mean_betweenness",
    "max_betweenness",
    "spectral_gap",
    "spectral_radius",
    "density",
    "degree_assortativity",
    "degree_skewness",
    "degree_kurtosis",
    "mean_in_degree",
    "mean_total_degree",
    "n_communities",
    "global_efficiency",
    "transitivity",
    "rich_club_k50",
]


def compute_correlations(metrics_csv=None, results_csv=None, task="MNIST"):
    """Compute Spearman correlations between graph metrics and BPU accuracy.

    Args:
        metrics_csv: path to graph metrics CSV
        results_csv: path to benchmark results CSV
        task: which benchmark task

    Returns:
        DataFrame with metric, spearman_rho, p_value, significant (Bonferroni)
    """
    if metrics_csv is None:
        metrics_csv = os.path.join(RESULTS_DIR, "graph_metrics.csv")
    if results_csv is None:
        results_csv = os.path.join(RESULTS_DIR, "all_results.csv")

    metrics_df = pd.read_csv(metrics_csv)
    results_df = pd.read_csv(results_csv)

    # Only biological BPUs
    bio_results = results_df[results_df["type"] == "biological"]
    bio_task = bio_results[bio_results["task"] == task]

    # Average accuracy across seeds
    avg_acc = bio_task.groupby("organism")["accuracy"].mean().reset_index()

    # Merge metrics with accuracy
    merged = metrics_df.merge(avg_acc, left_on="name", right_on="organism", how="inner")

    if len(merged) < 4:
        print(f"WARNING: Only {len(merged)} data points for correlation. Need at least 4.")
        return None

    print(f"\n{'='*60}")
    print(f"GRAPH METRIC CORRELATIONS: {task}")
    print(f"{'='*60}")
    print(f"Data points: {len(merged)} biological connectomes")

    correlations = []
    valid_tests = 0

    for col in METRIC_COLS:
        if col not in merged.columns:
            continue

        values = merged[col].dropna()
        if len(values) < 4:
            continue

        valid_idx = merged[col].notna()
        x = merged.loc[valid_idx, col].values.astype(float)
        y = merged.loc[valid_idx, "accuracy"].values.astype(float)

        if np.std(x) == 0 or np.std(y) == 0:
            continue

        rho, p = stats.spearmanr(x, y)
        valid_tests += 1

        correlations.append({
            "metric": col,
            "spearman_rho": rho,
            "p_value": p,
            "abs_rho": abs(rho),
            "n_datapoints": len(x),
            "task": task,
        })

    corr_df = pd.DataFrame(correlations)

    if len(corr_df) == 0:
        print("No valid correlations computed.")
        return None

    # Bonferroni correction
    n_tests = len(corr_df)
    corr_df["p_bonferroni"] = corr_df["p_value"] * n_tests
    corr_df["significant_bonferroni"] = corr_df["p_bonferroni"] < 0.05

    # Also FDR correction (Benjamini-Hochberg)
    sorted_p = corr_df.sort_values("p_value")
    sorted_p["rank"] = range(1, len(sorted_p) + 1)
    sorted_p["p_fdr"] = sorted_p["p_value"] * n_tests / sorted_p["rank"]
    sorted_p["significant_fdr"] = sorted_p["p_fdr"] < 0.05
    corr_df = sorted_p.sort_values("abs_rho", ascending=False)

    # Print results
    print(f"\nTop correlations (sorted by |rho|):")
    print(f"{'Metric':<25} {'rho':>8} {'p':>10} {'p_bonf':>10} {'Sig?':>6}")
    print("-" * 65)
    for _, row in corr_df.head(10).iterrows():
        sig = "***" if row["significant_bonferroni"] else ("*" if row["p_value"] < 0.05 else "")
        print(f"{row['metric']:<25} {row['spearman_rho']:>8.4f} {row['p_value']:>10.6f} "
              f"{row['p_bonferroni']:>10.6f} {sig:>6}")

    return corr_df


def compute_all_task_correlations(metrics_csv=None, results_csv=None):
    """Run correlation analysis across all benchmark tasks."""
    if results_csv is None:
        results_csv = os.path.join(RESULTS_DIR, "all_results.csv")

    results_df = pd.read_csv(results_csv)
    tasks = results_df["task"].unique()

    all_corrs = []
    for task in tasks:
        corr = compute_correlations(metrics_csv, results_csv, task)
        if corr is not None:
            all_corrs.append(corr)

    if all_corrs:
        combined = pd.concat(all_corrs, ignore_index=True)
        output = os.path.join(RESULTS_DIR, "graph_metric_correlations.csv")
        combined.to_csv(output, index=False)
        print(f"\nSaved all correlations to {output}")

        # Summary: which metrics are consistently top predictors?
        print(f"\n{'='*60}")
        print("CONSISTENCY ANALYSIS: Top predictors across all tasks")
        print(f"{'='*60}")
        avg_rho = combined.groupby("metric")["abs_rho"].mean().sort_values(ascending=False)
        for metric, mean_rho in avg_rho.head(5).items():
            print(f"  {metric}: mean |rho| = {mean_rho:.4f}")

        return combined

    return None


if __name__ == "__main__":
    compute_all_task_correlations()
