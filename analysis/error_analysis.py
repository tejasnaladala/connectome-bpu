"""error_analysis.py — Comprehensive statistical & error analysis of BPU experiments.

Computes: std deviation, SEM, 95% CI, coefficient of variation, bootstrap CIs,
effect sizes (Cohen's d, Hedge's g, Glass's delta), normality tests, variance
homogeneity, paired tests, bio-vs-control breakdowns, loss convergence analysis,
scaling gradient analysis, and per-organism error budgets.

Usage:
    python analysis/error_analysis.py [--csv PATH] [--format json|text]
"""
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import json
import os
import sys
import warnings
import io
warnings.filterwarnings("ignore")

# Fix Windows encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0.0


def hedges_g(x, y):
    d = cohens_d(x, y)
    n = len(x) + len(y)
    correction = 1 - 3 / (4 * n - 9) if n > 9 else 1.0
    return d * correction


def glass_delta(x, y):
    """Glass's delta using control group std as denominator."""
    ctrl_std = np.std(y, ddof=1)
    return (np.mean(x) - np.mean(y)) / ctrl_std if ctrl_std > 0 else 0.0


def bootstrap_ci(data, n_boot=10000, ci=0.95, stat_func=np.mean):
    """Bootstrap confidence interval."""
    boot_stats = np.array([stat_func(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return np.percentile(boot_stats, [alpha * 100, (1 - alpha) * 100])


def effect_size_label(d):
    d = abs(d)
    if d < 0.2: return "negligible"
    elif d < 0.5: return "small"
    elif d < 0.8: return "medium"
    else: return "large"


def run_analysis(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(RESULTS_DIR, "cab_v2_results.csv")

    df = pd.read_csv(csv_path)
    n_total = len(df)
    bio = df[df["type"] == "biological"]
    ctrl = df[df["type"] != "biological"]
    organisms = sorted(df["organism"].unique())
    ctrl_types = sorted(ctrl["type"].unique())
    tasks = sorted(df["task"].unique())

    print("=" * 80)
    print("COMPREHENSIVE ERROR ANALYSIS — BPU EXPERIMENTS")
    print("=" * 80)
    print(f"Total experiments: {n_total}")
    print(f"Organisms: {len(organisms)} | Control types: {len(ctrl_types)} | Tasks: {len(tasks)}")
    print(f"Seeds: {sorted(df['seed'].unique())}")

    # ═══════════════════════════════════════════════════════════════════
    # 1. DESCRIPTIVE STATISTICS (per organism, bio vs each control)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("1. DESCRIPTIVE STATISTICS BY ORGANISM")
    print(f"{'='*80}")

    organism_stats = []
    for org in organisms:
        org_bio = df[(df["organism"] == org) & (df["type"] == "biological")]
        org_ctrl = df[(df["organism"] == org) & (df["type"] != "biological")]
        n_neurons = org_bio["n_neurons"].iloc[0] if len(org_bio) > 0 else 0

        bio_acc = org_bio["accuracy"].values
        ctrl_acc = org_ctrl["accuracy"].values
        bio_loss = org_bio["final_loss"].values
        ctrl_loss = org_ctrl["final_loss"].values

        bio_mean = np.mean(bio_acc) if len(bio_acc) > 0 else np.nan
        bio_std = np.std(bio_acc, ddof=1) if len(bio_acc) > 1 else 0
        bio_sem = bio_std / np.sqrt(len(bio_acc)) if len(bio_acc) > 0 else np.nan
        bio_cv = (bio_std / bio_mean * 100) if bio_mean > 0 else np.nan
        bio_ci = stats.t.interval(0.95, len(bio_acc)-1, loc=bio_mean, scale=bio_sem) if len(bio_acc) > 1 else (np.nan, np.nan)
        bio_boot_ci = bootstrap_ci(bio_acc) if len(bio_acc) >= 3 else (np.nan, np.nan)

        ctrl_mean = np.mean(ctrl_acc) if len(ctrl_acc) > 0 else np.nan
        ctrl_std = np.std(ctrl_acc, ddof=1) if len(ctrl_acc) > 1 else 0
        ctrl_sem = ctrl_std / np.sqrt(len(ctrl_acc)) if len(ctrl_acc) > 0 else np.nan
        ctrl_cv = (ctrl_std / ctrl_mean * 100) if ctrl_mean > 0 else np.nan

        # Effect sizes
        d = cohens_d(bio_acc, ctrl_acc) if len(bio_acc) > 0 and len(ctrl_acc) > 0 else np.nan
        g = hedges_g(bio_acc, ctrl_acc) if len(bio_acc) > 0 and len(ctrl_acc) > 0 else np.nan
        delta = glass_delta(bio_acc, ctrl_acc) if len(bio_acc) > 0 and len(ctrl_acc) > 0 else np.nan

        # Loss analysis
        loss_d = cohens_d(-bio_loss, -ctrl_loss) if len(bio_loss) > 0 and len(ctrl_loss) > 0 else np.nan
        bio_loss_mean = np.mean(bio_loss) if len(bio_loss) > 0 else np.nan
        ctrl_loss_mean = np.mean(ctrl_loss) if len(ctrl_loss) > 0 else np.nan

        # Normality (Shapiro-Wilk)
        sw_bio = stats.shapiro(bio_acc) if len(bio_acc) >= 3 else (np.nan, np.nan)
        sw_ctrl = stats.shapiro(ctrl_acc) if len(ctrl_acc) >= 3 else (np.nan, np.nan)

        # t-test (Welch's)
        if len(bio_acc) >= 2 and len(ctrl_acc) >= 2:
            t_stat, t_p = stats.ttest_ind(bio_acc, ctrl_acc, equal_var=False)
            # Mann-Whitney U (non-parametric)
            u_stat, u_p = stats.mannwhitneyu(bio_acc, ctrl_acc, alternative="two-sided") if len(bio_acc) >= 3 else (np.nan, np.nan)
        else:
            t_stat, t_p, u_stat, u_p = np.nan, np.nan, np.nan, np.nan

        # Levene's test (variance homogeneity)
        if len(bio_acc) >= 2 and len(ctrl_acc) >= 2:
            lev_stat, lev_p = stats.levene(bio_acc, ctrl_acc)
        else:
            lev_stat, lev_p = np.nan, np.nan

        row = {
            "organism": org, "n_neurons": n_neurons,
            "bio_n": len(bio_acc), "ctrl_n": len(ctrl_acc),
            "bio_mean": bio_mean, "bio_std": bio_std, "bio_sem": bio_sem,
            "bio_cv_pct": bio_cv,
            "bio_ci95_lo": bio_ci[0], "bio_ci95_hi": bio_ci[1],
            "bio_boot_ci_lo": bio_boot_ci[0], "bio_boot_ci_hi": bio_boot_ci[1],
            "ctrl_mean": ctrl_mean, "ctrl_std": ctrl_std, "ctrl_sem": ctrl_sem,
            "ctrl_cv_pct": ctrl_cv,
            "delta_acc": bio_mean - ctrl_mean if not np.isnan(bio_mean) else np.nan,
            "delta_pct": (bio_mean - ctrl_mean) * 100 if not np.isnan(bio_mean) else np.nan,
            "cohens_d": d, "hedges_g": g, "glass_delta": delta,
            "effect_label": effect_size_label(d) if not np.isnan(d) else "N/A",
            "bio_loss_mean": bio_loss_mean, "ctrl_loss_mean": ctrl_loss_mean,
            "loss_delta": bio_loss_mean - ctrl_loss_mean if not np.isnan(bio_loss_mean) else np.nan,
            "loss_cohens_d": loss_d,
            "shapiro_bio_p": sw_bio[1], "shapiro_ctrl_p": sw_ctrl[1],
            "welch_t": t_stat, "welch_p": t_p,
            "mannwhitney_u": u_stat, "mannwhitney_p": u_p,
            "levene_stat": lev_stat, "levene_p": lev_p,
        }
        organism_stats.append(row)

        print(f"\n--- {org.upper()} ({n_neurons} neurons) ---")
        print(f"  Bio:  mean={bio_mean:.4f}  std={bio_std:.4f}  SEM={bio_sem:.4f}  CV={bio_cv:.2f}%  n={len(bio_acc)}")
        print(f"        95% CI: [{bio_ci[0]:.4f}, {bio_ci[1]:.4f}]  Bootstrap CI: [{bio_boot_ci[0]:.4f}, {bio_boot_ci[1]:.4f}]")
        print(f"  Ctrl: mean={ctrl_mean:.4f}  std={ctrl_std:.4f}  SEM={ctrl_sem:.4f}  CV={ctrl_cv:.2f}%  n={len(ctrl_acc)}")
        print(f"  Δacc: {(bio_mean - ctrl_mean)*100:+.3f}%  Cohen's d={d:+.3f} ({effect_size_label(d)})  Hedge's g={g:+.3f}  Glass's Δ={delta:+.3f}")
        print(f"  Loss: bio={bio_loss_mean:.5f}  ctrl={ctrl_loss_mean:.5f}  Δ={bio_loss_mean - ctrl_loss_mean:+.5f}  d(loss)={loss_d:+.3f}")
        print(f"  Normality: bio p={sw_bio[1]:.4f}  ctrl p={sw_ctrl[1]:.4f}  {'NORMAL' if sw_bio[1] > 0.05 and sw_ctrl[1] > 0.05 else 'NON-NORMAL'}")
        print(f"  Welch's t: t={t_stat:.3f}  p={t_p:.4f}  {'SIG' if t_p < 0.05 else 'NS'}")
        print(f"  Mann-Whitney U: U={u_stat:.0f}  p={u_p:.4f}  {'SIG' if u_p < 0.05 else 'NS'}" if not np.isnan(u_p) else "")
        print(f"  Levene's: F={lev_stat:.3f}  p={lev_p:.4f}  {'EQUAL VAR' if lev_p > 0.05 else 'UNEQUAL VAR'}")

    # ═══════════════════════════════════════════════════════════════════
    # 2. PER-CONTROL-TYPE BREAKDOWN
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("2. BIO vs EACH CONTROL TYPE (with effect sizes)")
    print(f"{'='*80}")

    ctrl_breakdown = []
    for org in organisms:
        org_bio = df[(df["organism"] == org) & (df["type"] == "biological")]["accuracy"].values
        for ct in ctrl_types:
            ct_acc = df[(df["organism"] == org) & (df["type"] == ct)]["accuracy"].values
            if len(org_bio) == 0 or len(ct_acc) == 0:
                continue
            d = cohens_d(org_bio, ct_acc)
            g = hedges_g(org_bio, ct_acc)
            bio_wins = np.mean(org_bio) > np.mean(ct_acc)
            if len(org_bio) >= 2 and len(ct_acc) >= 2:
                t_stat, t_p = stats.ttest_ind(org_bio, ct_acc, equal_var=False)
            else:
                t_stat, t_p = np.nan, np.nan

            row = {
                "organism": org, "control": ct,
                "bio_mean": np.mean(org_bio), "ctrl_mean": np.mean(ct_acc),
                "bio_std": np.std(org_bio, ddof=1), "ctrl_std": np.std(ct_acc, ddof=1),
                "delta": np.mean(org_bio) - np.mean(ct_acc),
                "cohens_d": d, "hedges_g": g,
                "effect": effect_size_label(d),
                "bio_wins": bio_wins,
                "welch_p": t_p
            }
            ctrl_breakdown.append(row)
            sig = "*" if t_p < 0.05 else ""
            winner = "BIO" if bio_wins else ct.upper()
            print(f"  {org:20s} vs {ct:20s}: Δ={row['delta']*100:+.3f}%  d={d:+.3f} ({effect_size_label(d):10s})  p={t_p:.4f}{sig}  → {winner}")

    # ═══════════════════════════════════════════════════════════════════
    # 3. ERROR BUDGET DECOMPOSITION
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("3. ERROR BUDGET DECOMPOSITION")
    print(f"{'='*80}")
    print("Variance sources: seed randomness vs topology vs organism")

    for org in organisms:
        org_data = df[df["organism"] == org]
        if len(org_data) < 3:
            continue

        total_var = np.var(org_data["accuracy"], ddof=1)
        seed_vars = []
        topo_vars = []

        # Within-type variance (seed effect)
        for t in org_data["type"].unique():
            t_data = org_data[org_data["type"] == t]["accuracy"].values
            if len(t_data) > 1:
                seed_vars.append(np.var(t_data, ddof=1))

        # Between-type variance (topology effect)
        type_means = org_data.groupby("type")["accuracy"].mean().values
        if len(type_means) > 1:
            topo_var = np.var(type_means, ddof=1)
        else:
            topo_var = 0

        mean_seed_var = np.mean(seed_vars) if seed_vars else 0
        seed_pct = mean_seed_var / total_var * 100 if total_var > 0 else 0
        topo_pct = topo_var / total_var * 100 if total_var > 0 else 0
        residual_pct = max(0, 100 - seed_pct - topo_pct)

        print(f"  {org:20s}: total_var={total_var:.8f}  seed={seed_pct:.1f}%  topology={topo_pct:.1f}%  residual={residual_pct:.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # 4. SCALING GRADIENT — EFFECT SIZE vs NEURON COUNT
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("4. SCALING GRADIENT: EFFECT SIZE vs NEURON COUNT")
    print(f"{'='*80}")
    print("Testing: does biological advantage GROW with neuron count?")

    scaling_data = []
    for row in organism_stats:
        if not np.isnan(row["cohens_d"]):
            scaling_data.append((row["n_neurons"], row["cohens_d"], row["organism"]))

    if len(scaling_data) >= 3:
        neurons = np.array([s[0] for s in scaling_data])
        d_values = np.array([s[1] for s in scaling_data])
        log_n = np.log10(neurons)

        slope, intercept, r, p, se = stats.linregress(log_n, d_values)
        rho, rho_p = stats.spearmanr(neurons, d_values)

        print(f"  Linear fit: d = {slope:.4f} * log10(N) + {intercept:.4f}")
        print(f"  Pearson r = {r:.4f}  p = {p:.4f}")
        print(f"  Spearman rho = {rho:.4f}  p = {rho_p:.4f}")
        print(f"  Interpretation: Effect size {'INCREASES' if slope > 0 else 'DECREASES'} with neuron count")

        for n, d, org in scaling_data:
            print(f"    {org:20s}: N={n:5d}  d={d:+.4f} ({effect_size_label(d)})")

        # Critical neuron count where d crosses 0 (biology starts winning)
        if slope > 0 and intercept < 0:
            crossover = 10 ** (-intercept / slope)
            print(f"\n  *** CROSSOVER POINT: biology starts winning at ~{crossover:.0f} neurons ***")

    # ═══════════════════════════════════════════════════════════════════
    # 5. LOSS CONVERGENCE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("5. LOSS CONVERGENCE ANALYSIS")
    print(f"{'='*80}")

    bio_losses = bio["final_loss"].values
    ctrl_losses = ctrl["final_loss"].values

    bio_loss_mean = np.mean(bio_losses)
    bio_loss_std = np.std(bio_losses, ddof=1)
    ctrl_loss_mean = np.mean(ctrl_losses)
    ctrl_loss_std = np.std(ctrl_losses, ddof=1)

    loss_d = cohens_d(-bio_losses, -ctrl_losses)  # negative because lower loss = better
    if len(bio_losses) >= 2 and len(ctrl_losses) >= 2:
        loss_t, loss_p = stats.ttest_ind(bio_losses, ctrl_losses, equal_var=False)
        loss_u, loss_u_p = stats.mannwhitneyu(bio_losses, ctrl_losses, alternative="two-sided")
    else:
        loss_t, loss_p, loss_u, loss_u_p = np.nan, np.nan, np.nan, np.nan

    print(f"  Bio loss:  mean={bio_loss_mean:.6f}  std={bio_loss_std:.6f}  median={np.median(bio_losses):.6f}")
    print(f"  Ctrl loss: mean={ctrl_loss_mean:.6f}  std={ctrl_loss_std:.6f}  median={np.median(ctrl_losses):.6f}")
    print(f"  Δ(loss): {bio_loss_mean - ctrl_loss_mean:+.6f}  ({'BIO LOWER' if bio_loss_mean < ctrl_loss_mean else 'CTRL LOWER'})")
    print(f"  Cohen's d (lower=better): {loss_d:+.3f} ({effect_size_label(loss_d)})")
    print(f"  Welch's t: t={loss_t:.3f}  p={loss_p:.6f}  {'SIGNIFICANT' if loss_p < 0.05 else 'NOT SIGNIFICANT'}")
    print(f"  Mann-Whitney U: U={loss_u:.0f}  p={loss_u_p:.6f}  {'SIGNIFICANT' if loss_u_p < 0.05 else 'NOT SIGNIFICANT'}")

    # Per-organism loss
    print(f"\n  Per-organism loss convergence:")
    for org in organisms:
        org_bio_loss = df[(df["organism"] == org) & (df["type"] == "biological")]["final_loss"].values
        org_ctrl_loss = df[(df["organism"] == org) & (df["type"] != "biological")]["final_loss"].values
        if len(org_bio_loss) > 0 and len(org_ctrl_loss) > 0:
            d_loss = cohens_d(-org_bio_loss, -org_ctrl_loss)
            winner = "BIO" if np.mean(org_bio_loss) < np.mean(org_ctrl_loss) else "CTRL"
            print(f"    {org:20s}: bio={np.mean(org_bio_loss):.5f}±{np.std(org_bio_loss, ddof=1):.5f}  "
                  f"ctrl={np.mean(org_ctrl_loss):.5f}±{np.std(org_ctrl_loss, ddof=1):.5f}  "
                  f"d={d_loss:+.3f}  → {winner}")

    # ═══════════════════════════════════════════════════════════════════
    # 6. TRAIN TIME ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("6. TRAINING TIME ANALYSIS")
    print(f"{'='*80}")

    bio_time = bio["train_time_sec"].values
    ctrl_time = ctrl["train_time_sec"].values
    print(f"  Bio:  mean={np.mean(bio_time):.1f}s  std={np.std(bio_time, ddof=1):.1f}s  median={np.median(bio_time):.1f}s")
    print(f"  Ctrl: mean={np.mean(ctrl_time):.1f}s  std={np.std(ctrl_time, ddof=1):.1f}s  median={np.median(ctrl_time):.1f}s")
    print(f"  Bio is {np.mean(bio_time)/np.mean(ctrl_time):.2f}x the ctrl training time")

    for org in organisms:
        org_bio_t = df[(df["organism"] == org) & (df["type"] == "biological")]["train_time_sec"].values
        org_ctrl_t = df[(df["organism"] == org) & (df["type"] != "biological")]["train_time_sec"].values
        if len(org_bio_t) > 0 and len(org_ctrl_t) > 0:
            ratio = np.mean(org_bio_t) / np.mean(org_ctrl_t)
            print(f"    {org:20s}: bio={np.mean(org_bio_t):.0f}s  ctrl={np.mean(org_ctrl_t):.0f}s  ratio={ratio:.2f}x")

    # ═══════════════════════════════════════════════════════════════════
    # 7. HYPOTHESIS SCOREBOARD
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("7. HYPOTHESIS SCOREBOARD (current data)")
    print(f"{'='*80}")

    # H1: Scaling law
    bio_by_org = bio.groupby("organism").agg(
        acc=("accuracy", "mean"), n=("n_neurons", "first")
    ).sort_values("n")
    h1_r, h1_p = stats.pearsonr(np.log10(bio_by_org["n"].values), bio_by_org["acc"].values) if len(bio_by_org) >= 3 else (np.nan, np.nan)
    h1_rho, h1_rho_p = stats.spearmanr(bio_by_org["n"].values, bio_by_org["acc"].values) if len(bio_by_org) >= 3 else (np.nan, np.nan)

    # H2: Biology > random
    overall_d = cohens_d(bio["accuracy"].values, ctrl["accuracy"].values)
    overall_t, overall_p = stats.ttest_ind(bio["accuracy"].values, ctrl["accuracy"].values, equal_var=False)

    # H5: Sexual dimorphism
    herm = df[(df["organism"] == "celegans_herm") & (df["type"] == "biological")]["accuracy"].values
    male = df[(df["organism"] == "celegans_male") & (df["type"] == "biological")]["accuracy"].values
    if len(herm) >= 2 and len(male) >= 2:
        h5_d = cohens_d(male, herm)
        h5_t, h5_p = stats.ttest_ind(male, herm, equal_var=False)
    else:
        h5_d, h5_t, h5_p = np.nan, np.nan, np.nan

    print(f"""
  H1 — SCALING LAW (performance ~ log(neuron count))
       Pearson r = {h1_r:.4f}  p = {h1_p:.4f}  Spearman rho = {h1_rho:.4f}  p = {h1_rho_p:.4f}
       Status: {'SUPPORTED' if h1_p < 0.05 and h1_r > 0.5 else 'TRENDING' if h1_r > 0.5 else 'NOT SUPPORTED'}

  H2 — BIOLOGY > RANDOM (biological topology confers advantage)
       Overall d = {overall_d:+.4f} ({effect_size_label(overall_d)})  t = {overall_t:.3f}  p = {overall_p:.4f}
       Bio mean = {np.mean(bio['accuracy'].values):.4f}  Ctrl mean = {np.mean(ctrl['accuracy'].values):.4f}
       Bio wins in {sum(1 for r in ctrl_breakdown if r['bio_wins'])}/{len(ctrl_breakdown)} pairwise comparisons
       Status: {'SUPPORTED' if overall_p < 0.05 and overall_d > 0.2 else 'NOT SUPPORTED (negligible effect on MNIST)'}

  H3 — GRAPH METRICS PREDICT PERFORMANCE (requires correlation analysis)
       Status: PENDING (need graph metrics for controls too)

  H4 — SUB-CIRCUIT SPECIALIZATION (requires adult Drosophila data)
       Status: PENDING (0/6 sub-circuits completed)

  H5 — SEXUAL DIMORPHISM
       Male mean = {np.mean(male):.4f}  Herm mean = {np.mean(herm):.4f}  d = {h5_d:+.4f}
       Welch's t = {h5_t:.3f}  p = {h5_p:.4f}
       Confound: neuron count (male=559, herm=419) — {'' if h5_p < 0.05 else 'NOT '}SIGNIFICANT
       Status: {'SUPPORTED but confounded' if h5_p < 0.05 else 'NOT SUPPORTED (yet)'}
""")

    # ═══════════════════════════════════════════════════════════════════
    # 8. OVERALL SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════
    print(f"{'='*80}")
    print("8. SUMMARY METRICS TABLE")
    print(f"{'='*80}")
    print(f"{'Organism':<22} {'N':>5} {'Bio μ':>8} {'Bio σ':>8} {'Bio SEM':>8} {'Ctrl μ':>8} {'Ctrl σ':>8} {'Δ%':>8} {'d':>7} {'Effect':>10} {'p':>8}")
    print("-" * 115)
    for row in organism_stats:
        print(f"{row['organism']:<22} {row['n_neurons']:>5} {row['bio_mean']:>8.4f} {row['bio_std']:>8.4f} {row['bio_sem']:>8.4f} "
              f"{row['ctrl_mean']:>8.4f} {row['ctrl_std']:>8.4f} {row['delta_pct']:>+8.3f} {row['cohens_d']:>+7.3f} "
              f"{row['effect_label']:>10} {row['welch_p']:>8.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # 9. SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════
    pd.DataFrame(organism_stats).to_csv(os.path.join(RESULTS_DIR, "error_analysis_organisms.csv"), index=False)
    pd.DataFrame(ctrl_breakdown).to_csv(os.path.join(RESULTS_DIR, "error_analysis_controls.csv"), index=False)

    # JSON summary
    summary = {
        "n_experiments": n_total,
        "n_organisms": len(organisms),
        "n_ctrl_types": len(ctrl_types),
        "n_tasks": len(tasks),
        "overall_bio_mean": float(np.mean(bio["accuracy"].values)),
        "overall_bio_std": float(np.std(bio["accuracy"].values, ddof=1)),
        "overall_ctrl_mean": float(np.mean(ctrl["accuracy"].values)),
        "overall_ctrl_std": float(np.std(ctrl["accuracy"].values, ddof=1)),
        "overall_cohens_d": float(overall_d),
        "overall_welch_p": float(overall_p),
        "h1_pearson_r": float(h1_r) if not np.isnan(h1_r) else None,
        "h1_p": float(h1_p) if not np.isnan(h1_p) else None,
        "h2_supported": bool(overall_p < 0.05 and overall_d > 0.2),
        "loss_bio_lower": bool(bio_loss_mean < ctrl_loss_mean),
        "loss_cohens_d": float(loss_d),
        "loss_p": float(loss_p) if not np.isnan(loss_p) else None,
    }
    with open(os.path.join(RESULTS_DIR, "error_analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: error_analysis_organisms.csv, error_analysis_controls.csv, error_analysis_summary.json")
    return summary


if __name__ == "__main__":
    csv_path = None
    if "--csv" in sys.argv:
        idx = sys.argv.index("--csv")
        csv_path = sys.argv[idx + 1]
    run_analysis(csv_path)
