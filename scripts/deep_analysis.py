import pandas as pd
import numpy as np
from scipy import stats
import os

from artifact_status import require_reportable_artifact

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
require_reportable_artifact(RESULTS_DIR)
CSV = os.path.join(RESULTS_DIR, 'all_results.csv')
df = pd.read_csv(CSV)

bio = df[df['type'] == 'biological']
ctrl = df[df['type'] != 'biological']

# Exclude Audio (trivial 100% task)
df_real = df[df['task'] != 'Audio_Simulated'].copy()
bio_real = df_real[df_real['type'] == 'biological']
ctrl_real = df_real[df_real['type'] != 'biological']

print("=" * 60)
print(f"  EXPERIMENT STATUS: {len(df)}/900 ({len(df)/900*100:.1f}%)")
print(f"  REAL TASKS (excl Audio): {len(df_real)}")
print("=" * 60)

print(f"\n--- HEADLINE METRICS (excl Audio) ---")
print(f"Bio acc:    {bio_real['accuracy'].mean():.4f}")
print(f"Ctrl acc:   {ctrl_real['accuracy'].mean():.4f}")
delta = bio_real['accuracy'].mean() - ctrl_real['accuracy'].mean()
print(f"Delta:      {delta:+.4f}")
print(f"Bio train:  {bio_real['train_time_sec'].mean():.2f}s")
print(f"Ctrl train: {ctrl_real['train_time_sec'].mean():.2f}s")
spd = (ctrl_real['train_time_sec'].mean() - bio_real['train_time_sec'].mean()) / ctrl_real['train_time_sec'].mean() * 100
print(f"Speed adv:  {spd:+.1f}%")

# Per-task breakdown
print(f"\n--- BY TASK (excl Audio) ---")
for task in sorted(df_real['task'].unique()):
    b = bio_real[bio_real['task'] == task]
    c = ctrl_real[ctrl_real['task'] == task]
    if len(b) > 0 and len(c) > 0:
        d = b['accuracy'].mean() - c['accuracy'].mean()
        spd_t = (c['train_time_sec'].mean() - b['train_time_sec'].mean()) / c['train_time_sec'].mean() * 100
        print(f"  {task:20s}: delta={d:+.4f}  speed={spd_t:+.1f}%  n={len(b)+len(c)}")

# Per-organism
print(f"\n--- PER-ORGANISM (excl Audio) ---")
df_real['org_group'] = df_real['name'].str.replace(
    r'_(erdos_renyi|barabasi_albert|watts_strogatz|degree_preserved)$', '', regex=True
)
for org in sorted(df_real['org_group'].unique()):
    b = df_real[(df_real['org_group'] == org) & (df_real['type'] == 'biological')]
    c = df_real[(df_real['org_group'] == org) & (df_real['type'] != 'biological')]
    if len(b) > 0 and len(c) > 0:
        d = b['accuracy'].mean() - c['accuracy'].mean()
        neurons = b['n_neurons'].iloc[0] if 'n_neurons' in b.columns else '?'
        print(f"  {org:35s}: delta={d:+.4f}  neurons={neurons}  n_bio={len(b)} n_ctrl={len(c)}")

# Control type ranking
print(f"\n--- CONTROL TYPE RANKING (excl Audio) ---")
for ct in ['biological', 'erdos_renyi', 'barabasi_albert', 'watts_strogatz', 'degree_preserved']:
    subset = df_real[df_real['type'] == ct]
    if len(subset) > 0:
        print(f"  {ct:25s}: acc={subset['accuracy'].mean():.4f}  train={subset['train_time_sec'].mean():.1f}s  n={len(subset)}")

# Cohen's d
pooled_std = np.sqrt((bio_real['accuracy'].var() + ctrl_real['accuracy'].var()) / 2)
cohens_d = (bio_real['accuracy'].mean() - ctrl_real['accuracy'].mean()) / pooled_std if pooled_std > 0 else 0
print(f"\nCohen's d (acc): {cohens_d:.4f}")

# Wilcoxon paired test
print(f"\n--- STATISTICAL TESTS ---")
paired_bio = []
paired_ctrl = []
pair_labels = []
for task in df_real['task'].unique():
    for org in df_real['org_group'].unique():
        b_vals = df_real[(df_real['org_group'] == org) & (df_real['type'] == 'biological') & (df_real['task'] == task)]['accuracy']
        c_vals = df_real[(df_real['org_group'] == org) & (df_real['type'] != 'biological') & (df_real['task'] == task)]['accuracy']
        if len(b_vals) > 0 and len(c_vals) > 0:
            paired_bio.append(b_vals.mean())
            paired_ctrl.append(c_vals.mean())
            pair_labels.append(f"{org}/{task}")

if len(paired_bio) > 5:
    diffs = [b - c for b, c in zip(paired_bio, paired_ctrl)]
    non_zero = [(b, c) for b, c in zip(paired_bio, paired_ctrl) if b != c]
    if len(non_zero) > 5:
        stat, p = stats.wilcoxon([b for b, c in non_zero], [c for b, c in non_zero])
        print(f"Wilcoxon signed-rank: stat={stat:.2f}, p={p:.6f}, n_pairs={len(non_zero)}")

    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    ties = sum(1 for d in diffs if d == 0)
    total_decided = wins + losses
    print(f"Pairwise: wins={wins} losses={losses} ties={ties} win_rate={wins/total_decided*100:.1f}%" if total_decided > 0 else "No decided pairs")

    # Show biggest wins and losses
    indexed = sorted(zip(diffs, pair_labels), key=lambda x: x[0])
    print(f"\nBiggest bio LOSSES:")
    for d, label in indexed[:5]:
        print(f"  {d:+.4f}  {label}")
    print(f"\nBiggest bio WINS:")
    for d, label in indexed[-5:]:
        print(f"  {d:+.4f}  {label}")

# Speed test
print(f"\n--- SPEED ANALYSIS (excl Audio) ---")
paired_bio_t = []
paired_ctrl_t = []
for task in df_real['task'].unique():
    for org in df_real['org_group'].unique():
        b_t = df_real[(df_real['org_group'] == org) & (df_real['type'] == 'biological') & (df_real['task'] == task)]['train_time_sec']
        c_t = df_real[(df_real['org_group'] == org) & (df_real['type'] != 'biological') & (df_real['task'] == task)]['train_time_sec']
        if len(b_t) > 0 and len(c_t) > 0:
            paired_bio_t.append(b_t.mean())
            paired_ctrl_t.append(c_t.mean())

faster_count = sum(1 for b, c in zip(paired_bio_t, paired_ctrl_t) if b < c)
slower_count = sum(1 for b, c in zip(paired_bio_t, paired_ctrl_t) if b > c)
print(f"Bio faster in {faster_count}/{len(paired_bio_t)} pairs ({faster_count/len(paired_bio_t)*100:.1f}%)")
mean_speedup = np.mean([(c - b) / c * 100 for b, c in zip(paired_bio_t, paired_ctrl_t)])
print(f"Mean speedup: {mean_speedup:+.1f}%")

# Completion gaps
print(f"\n--- COMPLETION GAPS ---")
df['org_group'] = df['name'].str.replace(
    r'_(erdos_renyi|barabasi_albert|watts_strogatz|degree_preserved)$', '', regex=True
)
pivot = df.groupby(['org_group', 'task']).size().unstack(fill_value=0)
missing = []
for org in pivot.index:
    for task in pivot.columns:
        if pivot.loc[org, task] < 15:
            missing.append((org, task, pivot.loc[org, task]))
missing.sort(key=lambda x: x[2])
print(f"Incomplete cells: {len(missing)}/60")
for org, task, n in missing[:15]:
    print(f"  {org:35s} x {task:20s}: {n}/15")
if len(missing) > 15:
    print(f"  ... and {len(missing) - 15} more")
