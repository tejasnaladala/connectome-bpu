"""scaling_law.py — Fit and test evolutionary scaling laws for BPU performance.

Tests H1: BPU performance increases monotonically with log(neuron count) across species.
Fits power-law, logarithmic, and linear models to accuracy vs. neuron count.

Usage:
    python analysis/scaling_law.py
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def power_law(x, a, b):
    """y = a * x^b"""
    return a * np.power(x, b)


def logarithmic(x, a, b):
    """y = a * log(x) + b"""
    return a * np.log10(x) + b


def fit_scaling_law(results_csv=None, task="MNIST"):
    """Fit scaling law to BPU performance data.

    Args:
        results_csv: path to results CSV (default: results/all_results.csv)
        task: which benchmark task to analyze

    Returns:
        dict with fit parameters, R-squared, p-values for each model
    """
    if results_csv is None:
        results_csv = os.path.join(RESULTS_DIR, "all_results.csv")

    df = pd.read_csv(results_csv)

    # Only biological BPUs
    bio = df[df["type"] == "biological"]
    bio_task = bio[bio["task"] == task]

    if len(bio_task) == 0:
        print(f"No biological results found for task={task}")
        return None

    # Average across seeds
    avg = bio_task.groupby("organism").agg(
        accuracy=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        n_neurons=("n_neurons", "first"),
        n_runs=("accuracy", "count")
    ).reset_index()

    avg = avg.sort_values("n_neurons")

    print(f"\n{'='*60}")
    print(f"SCALING LAW ANALYSIS: {task}")
    print(f"{'='*60}")
    print(f"\nOrganisms (sorted by neuron count):")
    for _, row in avg.iterrows():
        print(f"  {row['organism']}: {row['n_neurons']} neurons -> "
              f"{row['accuracy']:.4f} +/- {row['accuracy_std']:.4f} "
              f"({row['n_runs']} runs)")

    N = avg["n_neurons"].values.astype(float)
    acc = avg["accuracy"].values

    results = {"task": task, "n_species": len(avg)}

    # === Model 1: Log-linear (log(accuracy) ~ log(N)) ===
    log_N = np.log10(N)
    log_acc = np.log10(np.clip(acc, 1e-10, None))  # Clip to avoid log(0)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_N, log_acc)
    results["loglog_slope"] = slope
    results["loglog_intercept"] = intercept
    results["loglog_r_squared"] = r_value ** 2
    results["loglog_p_value"] = p_value
    results["loglog_std_err"] = std_err

    print(f"\n--- Log-Log Linear Fit ---")
    print(f"  log(accuracy) = {slope:.4f} * log(N) + {intercept:.4f}")
    print(f"  R^2 = {r_value**2:.4f}, p = {p_value:.6f}")

    # === Model 2: Logarithmic (accuracy ~ a*log(N) + b) ===
    try:
        popt, pcov = curve_fit(logarithmic, N, acc)
        predicted = logarithmic(N, *popt)
        ss_res = np.sum((acc - predicted) ** 2)
        ss_tot = np.sum((acc - np.mean(acc)) ** 2)
        r2_log = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        results["log_a"] = popt[0]
        results["log_b"] = popt[1]
        results["log_r_squared"] = r2_log

        print(f"\n--- Logarithmic Fit ---")
        print(f"  accuracy = {popt[0]:.6f} * log10(N) + {popt[1]:.4f}")
        print(f"  R^2 = {r2_log:.4f}")
    except Exception as e:
        print(f"\n--- Logarithmic Fit FAILED: {e} ---")
        results["log_r_squared"] = None

    # === Model 3: Power Law (accuracy ~ a * N^b) ===
    try:
        popt, pcov = curve_fit(power_law, N, acc, p0=[0.5, 0.1], maxfev=5000)
        predicted = power_law(N, *popt)
        ss_res = np.sum((acc - predicted) ** 2)
        ss_tot = np.sum((acc - np.mean(acc)) ** 2)
        r2_pow = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        results["power_a"] = popt[0]
        results["power_b"] = popt[1]
        results["power_r_squared"] = r2_pow

        print(f"\n--- Power Law Fit ---")
        print(f"  accuracy = {popt[0]:.6f} * N^{popt[1]:.4f}")
        print(f"  R^2 = {r2_pow:.4f}")
    except Exception as e:
        print(f"\n--- Power Law Fit FAILED: {e} ---")
        results["power_r_squared"] = None

    # === Monotonicity test (Spearman rank) ===
    rho, p_mono = stats.spearmanr(N, acc)
    results["spearman_rho"] = rho
    results["spearman_p"] = p_mono
    results["is_monotonic"] = rho > 0 and p_mono < 0.05

    print(f"\n--- Monotonicity Test (Spearman) ---")
    print(f"  rho = {rho:.4f}, p = {p_mono:.6f}")
    print(f"  H1 {'SUPPORTED' if results['is_monotonic'] else 'NOT SUPPORTED'}: "
          f"performance {'increases' if rho > 0 else 'decreases'} with neuron count")

    # === Best model selection ===
    r2_values = {
        "loglog": results.get("loglog_r_squared"),
        "logarithmic": results.get("log_r_squared"),
        "power_law": results.get("power_r_squared")
    }
    valid_r2 = {k: v for k, v in r2_values.items() if v is not None}
    if valid_r2:
        best = max(valid_r2, key=valid_r2.get)
        results["best_model"] = best
        results["best_r_squared"] = valid_r2[best]
        print(f"\n  Best fit model: {best} (R^2 = {valid_r2[best]:.4f})")

    return results


def fit_all_tasks(results_csv=None):
    """Fit scaling laws across all benchmark tasks."""
    if results_csv is None:
        results_csv = os.path.join(RESULTS_DIR, "all_results.csv")

    df = pd.read_csv(results_csv)
    tasks = df["task"].unique()

    all_fits = []
    for task in tasks:
        fit = fit_scaling_law(results_csv, task)
        if fit:
            all_fits.append(fit)

    # Save
    fits_df = pd.DataFrame(all_fits)
    output = os.path.join(RESULTS_DIR, "scaling_law_fits.csv")
    fits_df.to_csv(output, index=False)
    print(f"\nSaved scaling law fits to {output}")
    return fits_df


if __name__ == "__main__":
    fit_all_tasks()
