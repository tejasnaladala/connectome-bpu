import pandas as pd
import os

from artifact_status import require_reportable_artifact

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
require_reportable_artifact(RESULTS_DIR)
CSV = os.path.join(RESULTS_DIR, 'all_results.csv')
df = pd.read_csv(CSV)

bio = df[df['type'] == 'biological']
ctrl = df[df['type'] != 'biological']

print(f"Total: {len(df)}/900 = {len(df)/900*100:.1f}%")
print(f"Duplicates: {df.duplicated(subset=['name','task','seed']).sum()}")
print(f"Bio entries: {len(bio)}, Control entries: {len(ctrl)}")
print()
print(f"Bio acc:  {bio['accuracy'].mean():.4f}")
print(f"Ctrl acc: {ctrl['accuracy'].mean():.4f}")
print(f"Delta:    {bio['accuracy'].mean() - ctrl['accuracy'].mean():+.4f}")
print()
print(f"Bio train:  {bio['train_time_sec'].mean():.2f}s")
print(f"Ctrl train: {ctrl['train_time_sec'].mean():.2f}s")
spd = (ctrl['train_time_sec'].mean() - bio['train_time_sec'].mean()) / ctrl['train_time_sec'].mean() * 100
print(f"Speed adv:  {spd:+.1f}%")
print()

print("=== BY TASK ===")
for task in sorted(df['task'].unique()):
    b = bio[bio['task'] == task]['accuracy']
    c = ctrl[ctrl['task'] == task]['accuracy']
    n = len(df[df['task'] == task])
    delta = b.mean() - c.mean() if len(b) > 0 and len(c) > 0 else 0
    print(f"  {task:20s}: n={n:3d}  bio={b.mean():.4f}  ctrl={c.mean():.4f}  delta={delta:+.4f}")

print()
print("=== COMPLETION MATRIX (task x organism_group) ===")
df['org_group'] = df['name'].str.replace(r'_(erdos_renyi|barabasi_albert|watts_strogatz|degree_preserved)$', '', regex=True)
pivot = df.groupby(['org_group', 'task']).size().unstack(fill_value=0)
print(pivot.to_string())
print()

# Expected: each cell should be 5 (1 bio + 4 controls) x 3 seeds = 15
# But bio only has 1 net_type so it's 5 types x 3 seeds = 15 per cell
total_expected = len(pivot.index) * len(pivot.columns) * 15
print(f"Matrix cells filled: {(pivot > 0).sum().sum()}/{len(pivot.index) * len(pivot.columns)}")
