"""specialization.py — Test functional specialization of brain sub-circuits.

Tests H4: Do adult fly sub-circuits show task-specific computational advantages?
Tests H5: Does C. elegans sexual dimorphism affect BPU performance?

Usage:
    python analysis/specialization.py
"""
import pandas as pd
import numpy as np
from scipy import stats
import os
import sys
import itertools

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Expected specialization predictions (from design doc)
SPECIALIZATION_HYPOTHESES = {
    # (better_circuit, worse_circuit, task) — we predict better > worse
    "H4a": ("adult_optic_lobe_medulla", "adult_mushroom_body", "MNIST"),
    "H4b": ("adult_optic_lobe_medulla", "adult_mushroom_body", "CIFAR10"),
    "H4c": ("adult_mushroom_body", "adult_optic_lobe_medulla", "SequentialMNIST"),
    "H4d": ("adult_central_complex", "adult_antennal_lobe", "CartPole"),
}


def test_specialization(results_csv=None):
    """Test all specialization hypotheses.

    Returns:
        dict of hypothesis results
    """
    if results_csv is None:
        results_csv = os.path.join(RESULTS_DIR, "cab_v2_results.csv")

    df = pd.read_csv(results_csv)
    bio = df[df["type"] == "biological"]
    subcircuits = bio[bio["organism"].str.startswith("adult_")]

    if len(subcircuits) == 0:
        print("No sub-circuit results found.")
        return None

    print(f"\n{'='*60}")
    print("SUB-CIRCUIT SPECIALIZATION ANALYSIS")
    print(f"{'='*60}")

    # === 1. Rankings per task ===
    print("\n--- Sub-Circuit Rankings by Task ---")
    tasks = subcircuits["task"].unique()
    rankings = {}

    for task in sorted(tasks):
        task_data = subcircuits[subcircuits["task"] == task]
        avg = task_data.groupby("organism")["accuracy"].agg(["mean", "std", "count"]).reset_index()
        avg = avg.sort_values("mean", ascending=False)
        rankings[task] = avg

        print(f"\n  {task}:")
        for _, row in avg.iterrows():
            name_short = row["organism"].replace("adult_", "")
            print(f"    {name_short:<25} {row['mean']:.4f} +/- {row['std']:.4f} (n={row['count']})")

    # === 2. Hypothesis tests ===
    print(f"\n--- Hypothesis Tests ---")
    hypothesis_results = {}

    for h_name, (better, worse, task) in SPECIALIZATION_HYPOTHESES.items():
        task_data = subcircuits[subcircuits["task"] == task]
        better_acc = task_data[task_data["organism"] == better]["accuracy"].values
        worse_acc = task_data[task_data["organism"] == worse]["accuracy"].values

        if len(better_acc) == 0 or len(worse_acc) == 0:
            print(f"\n  {h_name}: SKIPPED (missing data for {better} or {worse})")
            hypothesis_results[h_name] = {"status": "skipped", "reason": "missing data"}
            continue

        # One-sided Mann-Whitney U test (better > worse)
        U, p_two = stats.mannwhitneyu(better_acc, worse_acc, alternative='greater')
        mean_diff = np.mean(better_acc) - np.mean(worse_acc)

        # Effect size (rank-biserial correlation)
        n1, n2 = len(better_acc), len(worse_acc)
        r_effect = 1 - (2 * U) / (n1 * n2)

        supported = p_two < 0.05 and mean_diff > 0

        result = {
            "status": "SUPPORTED" if supported else "NOT SUPPORTED",
            "better": better.replace("adult_", ""),
            "worse": worse.replace("adult_", ""),
            "task": task,
            "mean_better": float(np.mean(better_acc)),
            "mean_worse": float(np.mean(worse_acc)),
            "mean_diff": float(mean_diff),
            "U_statistic": float(U),
            "p_value": float(p_two),
            "effect_size_r": float(r_effect),
        }

        hypothesis_results[h_name] = result

        print(f"\n  {h_name}: {result['better']} > {result['worse']} on {task}")
        print(f"    Mean: {result['mean_better']:.4f} vs {result['mean_worse']:.4f} "
              f"(diff = {result['mean_diff']:+.4f})")
        print(f"    Mann-Whitney U = {U:.1f}, p = {p_two:.6f}, r = {r_effect:.4f}")
        print(f"    -> {result['status']}")

    # === 3. Overall specialization score ===
    print(f"\n--- Specialization Summary ---")

    # Compute specialization index: for each sub-circuit, how variable is its
    # ranking across tasks? High variance = generalist, low variance = specialist
    print("\n  Specialization Index (lower = more specialized):")
    for organism in subcircuits["organism"].unique():
        org_data = subcircuits[subcircuits["organism"] == organism]
        # Get rank per task
        ranks = []
        for task in tasks:
            task_data = subcircuits[subcircuits["task"] == task]
            avg = task_data.groupby("organism")["accuracy"].mean().sort_values(ascending=False)
            if organism in avg.index:
                rank = list(avg.index).index(organism) + 1
                ranks.append(rank)
        if ranks:
            spec_index = np.std(ranks)
            name_short = organism.replace("adult_", "")
            print(f"    {name_short:<25} rank_std = {spec_index:.2f} "
                  f"(ranks: {ranks})")

    return hypothesis_results


def test_sexual_dimorphism(results_csv=None):
    """Test H5: C. elegans male vs hermaphrodite BPU performance.

    Returns:
        dict of results per task
    """
    if results_csv is None:
        results_csv = os.path.join(RESULTS_DIR, "cab_v2_results.csv")

    df = pd.read_csv(results_csv)
    bio = df[df["type"] == "biological"]

    herm = bio[bio["organism"] == "celegans_herm"]
    male = bio[bio["organism"] == "celegans_male"]

    if len(herm) == 0 or len(male) == 0:
        print("Missing C. elegans data for dimorphism test.")
        return None

    print(f"\n{'='*60}")
    print("SEXUAL DIMORPHISM ANALYSIS: C. elegans")
    print(f"{'='*60}")

    results = {}
    tasks = set(herm["task"].unique()) & set(male["task"].unique())

    for task in sorted(tasks):
        h_acc = herm[herm["task"] == task]["accuracy"].values
        m_acc = male[male["task"] == task]["accuracy"].values

        if len(h_acc) == 0 or len(m_acc) == 0:
            continue

        U, p = stats.mannwhitneyu(m_acc, h_acc, alternative='two-sided')
        mean_diff = np.mean(m_acc) - np.mean(h_acc)

        result = {
            "task": task,
            "mean_male": float(np.mean(m_acc)),
            "mean_herm": float(np.mean(h_acc)),
            "diff": float(mean_diff),
            "p_value": float(p),
            "male_better": mean_diff > 0 and p < 0.05,
        }
        results[task] = result

        winner = "MALE" if result["male_better"] else ("HERM" if mean_diff < 0 and p < 0.05 else "NO DIFF")
        print(f"\n  {task}: male={result['mean_male']:.4f} vs herm={result['mean_herm']:.4f} "
              f"(diff={mean_diff:+.4f}, p={p:.4f}) -> {winner}")

    return results


if __name__ == "__main__":
    test_specialization()
    test_sexual_dimorphism()
