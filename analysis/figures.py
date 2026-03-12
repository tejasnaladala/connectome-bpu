"""figures.py — Generate all publication-quality figures for the paper.

Produces 7 figures:
1. Scaling law: Accuracy vs log(N) with fitted curves
2. Bio vs Random: Paired comparison for each organism
3. Graph metric heatmap: Correlation matrix of metrics vs performance
4. Sub-circuit specialization: Spider/radar chart
5. Sexual dimorphism: Male vs hermaphrodite comparison
6. Architecture diagram: BPU model schematic (manual)
7. Phylogenetic tree with performance overlay

Usage:
    python analysis/figures.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.optimize import curve_fit
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Publication style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Color scheme
COLORS = {
    'biological': '#2196F3',
    'erdos_renyi': '#9E9E9E',
    'barabasi_albert': '#FF9800',
    'watts_strogatz': '#4CAF50',
    'degree_preserved': '#9C27B0',
}

ORGANISM_LABELS = {
    'ciona': 'C. intestinalis',
    'celegans_herm': 'C. elegans (H)',
    'celegans_male': 'C. elegans (M)',
    'drosophila_larva': 'D. melanogaster (L)',
}

ORGANISM_SIZES = {
    'ciona': 177,
    'celegans_herm': 419,
    'celegans_male': 559,
    'drosophila_larva': 2952,
}


def load_results():
    """Load all results and graph metrics."""
    results = pd.read_csv(os.path.join(RESULTS_DIR, "all_results.csv"))
    try:
        metrics = pd.read_csv(os.path.join(RESULTS_DIR, "graph_metrics.csv"))
    except:
        metrics = None
    return results, metrics


def fig1_scaling_law(results, task="MNIST"):
    """Figure 1: Evolutionary scaling law."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    bio = results[(results["type"] == "biological") & (results["task"] == task)]
    if len(bio) == 0:
        print(f"No biological results for {task}")
        return

    # Average across seeds
    avg = bio.groupby("organism").agg(
        accuracy=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        n_neurons=("n_neurons", "first"),
    ).reset_index()

    # Only whole organisms for scaling law
    whole = avg[avg["organism"].isin(ORGANISM_SIZES.keys())]
    whole = whole.sort_values("n_neurons")

    N = whole["n_neurons"].values.astype(float)
    acc = whole["accuracy"].values
    err = whole["accuracy_std"].values

    # Panel A: Accuracy vs log(N) with fit
    ax = axes[0]
    ax.errorbar(N, acc, yerr=err, fmt='o-', color=COLORS['biological'],
                markersize=8, capsize=4, linewidth=2, label='Biological BPU')

    # Fit log curve
    try:
        def log_model(x, a, b):
            return a * np.log10(x) + b
        popt, _ = curve_fit(log_model, N, acc)
        x_fit = np.logspace(np.log10(N.min()*0.8), np.log10(N.max()*1.2), 100)
        ax.plot(x_fit, log_model(x_fit, *popt), '--', color='red', alpha=0.7,
                label=f'Fit: {popt[0]:.4f}·log₁₀(N) + {popt[1]:.3f}')
    except:
        pass

    # Add organism labels
    for _, row in whole.iterrows():
        label = ORGANISM_LABELS.get(row["organism"], row["organism"])
        ax.annotate(label, (row["n_neurons"], row["accuracy"]),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=8, fontstyle='italic')

    ax.set_xscale('log')
    ax.set_xlabel('Number of Neurons (log scale)')
    ax.set_ylabel(f'{task} Accuracy')
    ax.set_title('A) Evolutionary Scaling Law')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Panel B: Bio vs Controls for each organism
    ax = axes[1]
    organisms = whole["organism"].tolist()
    x_pos = np.arange(len(organisms))
    width = 0.15

    for i, org in enumerate(organisms):
        org_data = results[(results["organism"] == org) & (results["task"] == task)]
        bio_acc = org_data[org_data["type"] == "biological"]["accuracy"].mean()

        ax.bar(i - 2*width, bio_acc, width, color=COLORS['biological'],
               label='Biological' if i == 0 else None)

        for j, ctrl in enumerate(['erdos_renyi', 'barabasi_albert', 'watts_strogatz', 'degree_preserved']):
            ctrl_data = org_data[org_data["type"] == ctrl]
            if len(ctrl_data) > 0:
                ctrl_acc = ctrl_data["accuracy"].mean()
                ax.bar(i + (j-1)*width, ctrl_acc, width, color=COLORS[ctrl],
                       label=ctrl.replace('_', ' ').title() if i == 0 else None)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([ORGANISM_LABELS.get(o, o) for o in organisms],
                       rotation=20, ha='right', fontsize=8)
    ax.set_ylabel(f'{task} Accuracy')
    ax.set_title('B) Biological vs. Synthetic Controls')
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"fig1_scaling_law_{task}.png")
    fig.savefig(path)
    plt.close()
    print(f"Saved {path}")


def fig2_bio_advantage(results, task="MNIST"):
    """Figure 2: Biological advantage over controls."""
    fig, ax = plt.subplots(figsize=(8, 5))

    bio = results[(results["type"] == "biological") & (results["task"] == task)]
    organisms = sorted(bio["organism"].unique(),
                       key=lambda x: ORGANISM_SIZES.get(x, 999))

    for i, org in enumerate(organisms):
        org_data = results[(results["organism"] == org) & (results["task"] == task)]
        bio_accs = org_data[org_data["type"] == "biological"]["accuracy"].values
        bio_mean = bio_accs.mean()

        for ctrl in ['erdos_renyi', 'barabasi_albert', 'watts_strogatz', 'degree_preserved']:
            ctrl_accs = org_data[org_data["type"] == ctrl]["accuracy"].values
            if len(ctrl_accs) > 0:
                diff = bio_mean - ctrl_accs.mean()
                color = 'green' if diff > 0 else 'red'
                ax.scatter(i, diff * 100, c=COLORS[ctrl], s=100, zorder=3,
                           label=ctrl.replace('_', ' ').title() if i == 0 else None)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(range(len(organisms)))
    ax.set_xticklabels([ORGANISM_LABELS.get(o, o) for o in organisms],
                       rotation=20, ha='right')
    ax.set_ylabel('Biological Advantage (percentage points)')
    ax.set_title(f'Biological vs. Control Accuracy Difference ({task})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"fig2_bio_advantage_{task}.png")
    fig.savefig(path)
    plt.close()
    print(f"Saved {path}")


def fig3_metric_correlation(results, metrics, task="MNIST"):
    """Figure 3: Graph metric correlation heatmap."""
    if metrics is None:
        print("No graph metrics available")
        return

    bio = results[(results["type"] == "biological") & (results["task"] == task)]
    avg_acc = bio.groupby("organism")["accuracy"].mean().reset_index()
    merged = metrics.merge(avg_acc, left_on="name", right_on="organism", how="inner")

    if len(merged) < 3:
        print(f"Only {len(merged)} data points, need at least 3")
        return

    metric_cols = [c for c in metrics.columns if c not in
                   ['name', 'n_neurons', 'n_edges', 'n_communities',
                    'rich_club_k25', 'rich_club_k75', 'largest_cc_fraction']]
    metric_cols = [c for c in metric_cols if c in merged.columns and merged[c].notna().sum() >= 3]

    correlations = {}
    for col in metric_cols:
        valid = merged[col].notna()
        if valid.sum() >= 3:
            rho, p = stats.spearmanr(merged.loc[valid, col], merged.loc[valid, "accuracy"])
            correlations[col] = {"rho": rho, "p": p}

    if not correlations:
        print("No valid correlations")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    corr_df = pd.DataFrame(correlations).T.sort_values("rho", ascending=True)

    colors = ['red' if r < 0 else 'steelblue' for r in corr_df['rho']]
    bars = ax.barh(range(len(corr_df)), corr_df['rho'], color=colors, alpha=0.8)

    # Mark significant
    for i, (_, row) in enumerate(corr_df.iterrows()):
        if row['p'] < 0.05:
            ax.text(row['rho'] + 0.02 * np.sign(row['rho']), i, '*',
                    fontsize=14, va='center', fontweight='bold')

    ax.set_yticks(range(len(corr_df)))
    ax.set_yticklabels(corr_df.index, fontsize=8)
    ax.set_xlabel('Spearman Correlation (ρ)')
    ax.set_title(f'Graph Metrics vs. {task} Accuracy')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"fig3_metric_correlation_{task}.png")
    fig.savefig(path)
    plt.close()
    print(f"Saved {path}")


def fig4_subcircuit_specialization(results):
    """Figure 4: Sub-circuit performance across tasks."""
    subcircuit_data = results[results["organism"].str.startswith("adult_")]
    if len(subcircuit_data) == 0:
        print("No sub-circuit data available")
        return

    bio_sub = subcircuit_data[subcircuit_data["type"] == "biological"]
    tasks = bio_sub["task"].unique()
    organisms = bio_sub["organism"].unique()

    fig, ax = plt.subplots(figsize=(10, 6))

    avg = bio_sub.groupby(["organism", "task"])["accuracy"].mean().reset_index()
    pivot = avg.pivot(index="organism", columns="task", values="accuracy")

    pivot.plot(kind="bar", ax=ax, width=0.8, colormap="Set2")
    ax.set_ylabel("Accuracy")
    ax.set_title("Sub-Circuit Functional Specialization")
    ax.legend(title="Task", bbox_to_anchor=(1.05, 1))
    ax.set_xticklabels([x.replace("adult_", "").replace("_", " ").title()
                        for x in pivot.index], rotation=30, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_specialization.png")
    fig.savefig(path)
    plt.close()
    print(f"Saved {path}")


def fig5_sexual_dimorphism(results, task="MNIST"):
    """Figure 5: C. elegans male vs hermaphrodite."""
    herm = results[(results["organism"] == "celegans_herm") & (results["task"] == task)]
    male = results[(results["organism"] == "celegans_male") & (results["task"] == task)]

    if len(herm) == 0 or len(male) == 0:
        print("Missing C. elegans data")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    types = ['biological', 'erdos_renyi', 'barabasi_albert', 'watts_strogatz', 'degree_preserved']
    x = np.arange(len(types))
    width = 0.35

    herm_vals = [herm[herm["type"] == t]["accuracy"].mean() for t in types]
    male_vals = [male[male["type"] == t]["accuracy"].mean() for t in types]

    ax.bar(x - width/2, herm_vals, width, label='Hermaphrodite (419 neurons)',
           color='#E91E63', alpha=0.8)
    ax.bar(x + width/2, male_vals, width, label='Male (559 neurons)',
           color='#2196F3', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('_', '\n') for t in types], fontsize=8)
    ax.set_ylabel(f'{task} Accuracy')
    ax.set_title('C. elegans Sexual Dimorphism: Male vs Hermaphrodite')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"fig5_dimorphism_{task}.png")
    fig.savefig(path)
    plt.close()
    print(f"Saved {path}")


def fig6_multi_task_heatmap(results):
    """Figure 6: Heatmap of all organisms × all tasks."""
    bio = results[results["type"] == "biological"]
    if len(bio) == 0:
        return

    avg = bio.groupby(["organism", "task"])["accuracy"].mean().reset_index()
    pivot = avg.pivot(index="organism", columns="task", values="accuracy")

    # Sort by neuron count
    order = sorted(pivot.index, key=lambda x: ORGANISM_SIZES.get(x, 9999))
    pivot = pivot.loc[[o for o in order if o in pivot.index]]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    labels = [ORGANISM_LABELS.get(o, o.replace('adult_', '').replace('_', ' ').title())
              for o in pivot.index]
    ax.set_yticklabels(labels, fontsize=9)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha='center', va='center', fontsize=8)

    plt.colorbar(im, ax=ax, label='Accuracy')
    ax.set_title('BPU Performance: All Organisms × All Tasks')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig6_heatmap.png")
    fig.savefig(path)
    plt.close()
    print(f"Saved {path}")


def generate_all_figures():
    """Generate all figures."""
    print("=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)

    results, metrics = load_results()
    tasks = results["task"].unique()

    for task in tasks:
        fig1_scaling_law(results, task)
        fig2_bio_advantage(results, task)
        fig5_sexual_dimorphism(results, task)
        if metrics is not None:
            fig3_metric_correlation(results, metrics, task)

    fig4_subcircuit_specialization(results)
    fig6_multi_task_heatmap(results)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_figures()
