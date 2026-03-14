"""Full statistical report with std deviation, error analysis, confidence intervals."""
import pandas as pd
import numpy as np
from scipy import stats
import warnings, sys, io
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv('results/all_results.csv')
controls = ['erdos_renyi', 'barabasi_albert', 'watts_strogatz', 'degree_preserved']
bio = df[df['type'] == 'biological']
ctrl = df[df['type'] != 'biological']

print('=' * 85)
print(f'  FULL STATISTICAL REPORT — {len(df)} EXPERIMENTS ({bio.shape[0]} bio, {ctrl.shape[0]} ctrl)')
print('=' * 85)

# ── 1. COMPLETION ──
print(f'\n  Organisms: {df["organism"].nunique()} | Seeds: {sorted(df["seed"].unique())} | Task: MNIST')
for org in sorted(df['organism'].unique(), key=lambda x: df[df['organism']==x]['n_neurons'].iloc[0]):
    sub = df[df['organism'] == org]
    print(f'    {org:<25} N={sub["n_neurons"].iloc[0]:>5} | {len(sub)} exps ({sub["seed"].nunique()} seeds × {sub["type"].nunique()} types)')

# ── 2. GLOBAL ACCURACY ──
print(f'\n{"─" * 85}')
print('  ACCURACY — Global')
print(f'{"─" * 85}')
for label, d in [('Bio', bio), ('Ctrl', ctrl)]:
    se = d['accuracy'].sem()
    ci = 1.96 * se
    m = d['accuracy'].mean()
    print(f'  {label:4s}  μ={m:.6f}  σ={d["accuracy"].std():.6f}  SE={se:.6f}  '
          f'95%CI=[{m-ci:.6f},{m+ci:.6f}]  min={d["accuracy"].min():.4f}  max={d["accuracy"].max():.4f}  n={len(d)}')

diff = bio['accuracy'].mean() - ctrl['accuracy'].mean()
t_s, p_s = stats.ttest_ind(bio['accuracy'], ctrl['accuracy'], equal_var=False)
ps = np.sqrt((bio['accuracy'].std()**2 + ctrl['accuracy'].std()**2) / 2)
d_acc = diff / ps
u, p_u = stats.mannwhitneyu(bio['accuracy'], ctrl['accuracy'], alternative='two-sided')
diff_se = np.sqrt(bio['accuracy'].sem()**2 + ctrl['accuracy'].sem()**2)
ci_lo = diff - 1.96 * diff_se
ci_hi = diff + 1.96 * diff_se
print(f'  Δ={diff*100:+.4f}%  d={d_acc:+.4f}  Welch t={t_s:.4f} p={p_s:.6f}  '
      f'U={u:.0f} p={p_u:.6f}  95%CI_Δ=[{ci_lo*100:+.4f}%,{ci_hi*100:+.4f}%] '
      f'{"INCLUDES 0" if ci_lo <= 0 <= ci_hi else "EXCLUDES 0"}')

# ── 3. GLOBAL LOSS ──
print(f'\n{"─" * 85}')
print('  FINAL LOSS — Global')
print(f'{"─" * 85}')
for label, d in [('Bio', bio), ('Ctrl', ctrl)]:
    print(f'  {label:4s}  μ={d["final_loss"].mean():.6f}  σ={d["final_loss"].std():.6f}  '
          f'min={d["final_loss"].min():.6f}  max={d["final_loss"].max():.6f}')
t_l, p_l = stats.ttest_ind(bio['final_loss'], ctrl['final_loss'], equal_var=False)
psl = np.sqrt((bio['final_loss'].std()**2 + ctrl['final_loss'].std()**2) / 2)
d_loss = (bio['final_loss'].mean() - ctrl['final_loss'].mean()) / psl
print(f'  Δ_loss={bio["final_loss"].mean()-ctrl["final_loss"].mean():.6f}  d={d_loss:+.4f}  '
      f'Welch t={t_l:.4f}  p={p_l:.6f}  {"BIO BETTER (lower loss)" if bio["final_loss"].mean() < ctrl["final_loss"].mean() else "CTRL BETTER"}')

# ── 4. PER-ORGANISM ──
print(f'\n{"─" * 85}')
print('  PER-ORGANISM ANALYSIS')
print(f'{"─" * 85}')
print(f'  {"Organism":<25} {"N":>5} {"Bio μ±σ":>16} {"Ctrl μ±σ":>16} {"Δ%":>8} {"d_acc":>7} {"p_acc":>9} {"d_loss":>7} {"p_loss":>9}')
for org in sorted(df['organism'].unique(), key=lambda x: df[df['organism']==x]['n_neurons'].iloc[0]):
    sub = df[df['organism'] == org]
    b = sub[sub['type'] == 'biological']['accuracy']
    c = sub[sub['type'] != 'biological']['accuracy']
    bl = sub[sub['type'] == 'biological']['final_loss']
    cl = sub[sub['type'] != 'biological']['final_loss']
    n = sub['n_neurons'].iloc[0]
    delta = (b.mean() - c.mean()) * 100
    ps_a = np.sqrt((b.std()**2 + c.std()**2) / 2) if b.std() > 0 else 1e-9
    d_a = (b.mean() - c.mean()) / ps_a
    _, p_a = stats.ttest_ind(b, c, equal_var=False) if len(b) > 1 else (0, 1)
    ps_l = np.sqrt((bl.std()**2 + cl.std()**2) / 2) if bl.std() > 0 else 1e-9
    d_l = (bl.mean() - cl.mean()) / ps_l
    _, p_ll = stats.ttest_ind(bl, cl, equal_var=False) if len(bl) > 1 else (0, 1)
    print(f'  {org:<25} {n:>5} {b.mean():.4f}±{b.std():.4f} {c.mean():.4f}±{c.std():.4f} '
          f'{delta:>+7.3f} {d_a:>+6.3f} {p_a:>9.5f} {d_l:>+6.3f} {p_ll:>9.5f}')

# ── 5. PER-CONTROL TYPE ──
print(f'\n{"─" * 85}')
print('  BIO vs EACH CONTROL TYPE (pooled across organisms)')
print(f'{"─" * 85}')
for ct in controls:
    c = df[df['type'] == ct]['accuracy']
    delta = (bio['accuracy'].mean() - c.mean()) * 100
    ps_a = np.sqrt((bio['accuracy'].std()**2 + c.std()**2) / 2)
    d_a = (bio['accuracy'].mean() - c.mean()) / ps_a
    _, p_a = stats.ttest_ind(bio['accuracy'], c, equal_var=False)
    cl = df[df['type'] == ct]['final_loss']
    ps_l = np.sqrt((bio['final_loss'].std()**2 + cl.std()**2) / 2)
    d_l = (bio['final_loss'].mean() - cl.mean()) / ps_l
    _, p_ll = stats.ttest_ind(bio['final_loss'], cl, equal_var=False)
    print(f'  vs {ct:<22} acc_Δ={delta:>+7.4f}%  acc_d={d_a:>+6.4f}  p={p_a:.5f}  |  loss_d={d_l:>+6.4f}  p={p_ll:.5f}')

# ── 6. EFFECT SIZE MATRIX ──
print(f'\n{"─" * 85}')
print('  EFFECT SIZE MATRIX (Cohen d: bio - ctrl, + = bio wins)')
print(f'{"─" * 85}')
print(f'  {"Organism":<25} {"ER":>8} {"BA":>8} {"WS":>8} {"DP":>8} {"ALL":>8}')
for org in sorted(df['organism'].unique(), key=lambda x: df[df['organism']==x]['n_neurons'].iloc[0]):
    sub = df[df['organism'] == org]
    b = sub[sub['type'] == 'biological']['accuracy']
    ds = []
    for ct in controls:
        c = sub[sub['type'] == ct]['accuracy']
        ps_a = np.sqrt((b.std()**2 + c.std()**2) / 2) if b.std() > 0 else 1e-9
        ds.append((b.mean() - c.mean()) / ps_a)
    ac = sub[sub['type'] != 'biological']['accuracy']
    ps_all = np.sqrt((b.std()**2 + ac.std()**2) / 2) if b.std() > 0 else 1e-9
    d_all = (b.mean() - ac.mean()) / ps_all
    print(f'  {org:<25} {ds[0]:>+7.3f} {ds[1]:>+7.3f} {ds[2]:>+7.3f} {ds[3]:>+7.3f} {d_all:>+7.3f}')

# ── 7. EFFECT SIZE GRADIENT (KEY FINDING) ──
print(f'\n{"─" * 85}')
print('  EFFECT SIZE GRADIENT — Cohen d vs log(N)')
print(f'{"─" * 85}')
gradient = []
for org in sorted(df['organism'].unique(), key=lambda x: df[df['organism']==x]['n_neurons'].iloc[0]):
    sub = df[df['organism'] == org]
    b = sub[sub['type'] == 'biological']['accuracy']
    c = sub[sub['type'] != 'biological']['accuracy']
    n = sub['n_neurons'].iloc[0]
    ps_a = np.sqrt((b.std()**2 + c.std()**2) / 2) if b.std() > 0 else 1e-9
    d_a = (b.mean() - c.mean()) / ps_a
    bl = sub[sub['type'] == 'biological']['final_loss']
    cl = sub[sub['type'] != 'biological']['final_loss']
    ps_l = np.sqrt((bl.std()**2 + cl.std()**2) / 2) if bl.std() > 0 else 1e-9
    d_l = (bl.mean() - cl.mean()) / ps_l
    gradient.append((n, org, d_a, d_l))
    print(f'  N={n:>5} {org:<25} acc_d={d_a:>+7.4f}  loss_d={d_l:>+7.4f}')

ns = np.array([g[0] for g in gradient])
ds_acc = np.array([g[2] for g in gradient])
ds_loss = np.array([g[3] for g in gradient])
if len(ns) > 2:
    r_g, p_g = stats.pearsonr(np.log10(ns), ds_acc)
    r_gl, p_gl = stats.pearsonr(np.log10(ns), ds_loss)
    print(f'\n  Gradient r(log(N), acc_d) = {r_g:+.4f}  p={p_g:.6f}')
    print(f'  Gradient r(log(N), loss_d) = {r_gl:+.4f}  p={p_gl:.6f}')
    # crossover estimate
    slope, intercept, _, _, _ = stats.linregress(np.log10(ns), ds_acc)
    if slope != 0:
        crossover_logn = -intercept / slope
        crossover_n = 10 ** crossover_logn
        print(f'  Crossover (d=0): N ≈ {crossover_n:.0f} neurons (bio starts winning above this)')

# ── 8. SCALING LAW (H1) ──
print(f'\n{"─" * 85}')
print('  H1: SCALING LAW — log(N) vs accuracy')
print(f'{"─" * 85}')
bio_means = bio.groupby('n_neurons')['accuracy'].agg(['mean', 'std', 'count', 'sem']).sort_index()
for n, row in bio_means.iterrows():
    ci = 1.96 * row['sem']
    print(f'  N={n:>5} → {row["mean"]:.6f} ± {row["std"]:.6f} (SE={row["sem"]:.6f}, 95%CI=[{row["mean"]-ci:.6f},{row["mean"]+ci:.6f}], n={int(row["count"])})')
log_n = np.log10(bio_means.index.values.astype(float))
accs = bio_means['mean'].values
r_h1, p_h1 = stats.pearsonr(log_n, accs)
sl, ic, rv, pv, se = stats.linregress(log_n, accs)
print(f'  Bio:  r={r_h1:.4f}  p={p_h1:.6f}  slope={sl:.6f}  R²={rv**2:.4f}')

ctrl_means = ctrl.groupby('n_neurons')['accuracy'].agg(['mean', 'std']).sort_index()
log_nc = np.log10(ctrl_means.index.values.astype(float))
r_c, p_c = stats.pearsonr(log_nc, ctrl_means['mean'].values)
sl_c, _, rv_c, _, _ = stats.linregress(log_nc, ctrl_means['mean'].values)
print(f'  Ctrl: r={r_c:.4f}  p={p_c:.6f}  slope={sl_c:.6f}  R²={rv_c**2:.4f}')
print(f'  Bio slope / Ctrl slope = {sl/sl_c:.4f}x' if sl_c != 0 else '  Ctrl slope = 0')

# ── 9. ERROR ANALYSIS ──
print(f'\n{"─" * 85}')
print('  ERROR ANALYSIS — Reproducibility')
print(f'{"─" * 85}')
print(f'  {"Organism":<25} {"CV%":>8} {"Range%":>8} {"IQR%":>8} {"Seeds":>15}')
for org in sorted(df['organism'].unique(), key=lambda x: df[df['organism']==x]['n_neurons'].iloc[0]):
    sb = df[(df['organism'] == org) & (df['type'] == 'biological')]
    if len(sb) > 1:
        cv = sb['accuracy'].std() / sb['accuracy'].mean() * 100
        rng = (sb['accuracy'].max() - sb['accuracy'].min()) * 100
        q1, q3 = sb['accuracy'].quantile(0.25), sb['accuracy'].quantile(0.75)
        iqr = (q3 - q1) * 100
        print(f'  {org:<25} {cv:>7.4f} {rng:>7.3f} {iqr:>7.4f} {str(sorted(sb["seed"].unique())):>15}')

print(f'\n  Cross-seed consistency (bio):')
for seed in sorted(df['seed'].unique()):
    s = bio[bio['seed'] == seed]
    print(f'    Seed {seed}: μ={s["accuracy"].mean():.6f}  σ={s["accuracy"].std():.6f}  n={len(s)}')

# ── 10. PAIRWISE WIN/LOSS ──
print(f'\n{"─" * 85}')
print('  PAIRWISE: Bio wins vs controls (per organism×seed)')
print(f'{"─" * 85}')
wins, losses, ties = 0, 0, 0
for org in df['organism'].unique():
    for seed in df['seed'].unique():
        sub = df[(df['organism'] == org) & (df['seed'] == seed)]
        b = sub[sub['type'] == 'biological']['accuracy']
        if len(b) == 0:
            continue
        bv = b.values[0]
        for _, row in sub[sub['type'] != 'biological'].iterrows():
            if bv > row['accuracy']:
                wins += 1
            elif bv < row['accuracy']:
                losses += 1
            else:
                ties += 1
total = wins + losses + ties
print(f'  Bio WINS:  {wins}/{total} ({wins/total*100:.1f}%)')
print(f'  Bio LOSES: {losses}/{total} ({losses/total*100:.1f}%)')
print(f'  Ties:      {ties}/{total} ({ties/total*100:.1f}%)')
if wins + losses > 0:
    bp = stats.binomtest(wins, wins + losses, 0.5).pvalue
    print(f'  Binomial test (H0: 50/50): p={bp:.6f}')

# ── 11. ANOVA + KRUSKAL-WALLIS ──
print(f'\n{"─" * 85}')
print('  ANOVA & NON-PARAMETRIC TESTS')
print(f'{"─" * 85}')
groups = [df[df['type'] == t]['accuracy'].values for t in ['biological'] + controls]
f_stat, p_f = stats.f_oneway(*groups)
eta_sq = (f_stat * 4) / (f_stat * 4 + len(df) - 5)
print(f'  One-way ANOVA:    F(4,{len(df)-5})={f_stat:.4f}  p={p_f:.6f}  η²={eta_sq:.6f}')
h_stat, p_kw = stats.kruskal(*groups)
print(f'  Kruskal-Wallis:   H={h_stat:.4f}  p={p_kw:.6f}')

# ── 12. NORMALITY + HOMOGENEITY ──
print(f'\n{"─" * 85}')
print('  DISTRIBUTIONAL TESTS')
print(f'{"─" * 85}')
for label, data in [('Bio acc', bio['accuracy']), ('Ctrl acc', ctrl['accuracy']),
                     ('Bio loss', bio['final_loss']), ('Ctrl loss', ctrl['final_loss'])]:
    w, p = stats.shapiro(data) if len(data) >= 3 else (0, 1)
    print(f'  {label:<10} Shapiro-Wilk W={w:.4f}  p={p:.4f}  {"NORMAL" if p > 0.05 else "NON-NORMAL"}')
lev, p_lev = stats.levene(bio['accuracy'], ctrl['accuracy'])
print(f'  Levene (acc):  F={lev:.4f}  p={p_lev:.6f}  {"EQUAL var" if p_lev > 0.05 else "UNEQUAL var"}')
lev_l, p_lev_l = stats.levene(bio['final_loss'], ctrl['final_loss'])
print(f'  Levene (loss): F={lev_l:.4f}  p={p_lev_l:.6f}  {"EQUAL var" if p_lev_l > 0.05 else "UNEQUAL var"}')

# ── 13. SEXUAL DIMORPHISM (H5) ──
print(f'\n{"─" * 85}')
print('  H5: SEXUAL DIMORPHISM')
print(f'{"─" * 85}')
male = df[(df['organism'] == 'celegans_male') & (df['type'] == 'biological')]['accuracy']
herm = df[(df['organism'] == 'celegans_herm') & (df['type'] == 'biological')]['accuracy']
if len(male) > 0 and len(herm) > 0:
    delta_h5 = (male.mean() - herm.mean()) * 100
    ps_h5 = np.sqrt((male.std()**2 + herm.std()**2) / 2) if male.std() > 0 else 1e-9
    d_h5 = (male.mean() - herm.mean()) / ps_h5
    t_h5, p_h5 = stats.ttest_ind(male, herm, equal_var=False)
    print(f'  Male:  μ={male.mean():.6f} ± {male.std():.6f}  (559n, n={len(male)})')
    print(f'  Herm:  μ={herm.mean():.6f} ± {herm.std():.6f}  (419n, n={len(herm)})')
    print(f'  Δ={delta_h5:+.4f}%  d={d_h5:+.4f}  p={p_h5:.4f}  CONFOUNDED by N (559 vs 419)')

# ── 14. TRAIN TIME ──
print(f'\n{"─" * 85}')
print('  TRAINING TIME')
print(f'{"─" * 85}')
total_sec = df['train_time_sec'].sum()
avg = df['train_time_sec'].mean()
remaining = 900 - len(df)
print(f'  Total compute: {total_sec:.0f}s ({total_sec/3600:.1f}h)')
print(f'  Avg/experiment: {avg:.0f}s')
print(f'  Bio avg:  {bio["train_time_sec"].mean():.0f}s ± {bio["train_time_sec"].std():.0f}s')
print(f'  Ctrl avg: {ctrl["train_time_sec"].mean():.0f}s ± {ctrl["train_time_sec"].std():.0f}s')
print(f'  Remaining: {remaining} exps ≈ {remaining*avg/3600:.1f}h')

# ── 15. VERDICT ──
print(f'\n{"=" * 85}')
print('  VERDICT')
print(f'{"=" * 85}')
print(f'  H1 (Scaling Law):        r={r_h1:.4f}  {"✓ SUPPORTED" if r_h1 > 0.7 else "~ WEAK"}')
print(f'  H2 (Bio > Random):       d={d_acc:+.4f} {"✓ SUPPORTED" if d_acc > 0.2 else "✗ NOT SUPPORTED (MNIST ceiling)"}')
print(f'  Loss Convergence:        d={d_loss:+.4f} p={p_l:.6f} {"✓ BIO WINS" if d_loss < 0 and p_l < 0.05 else "~ CHECK"}')
if len(ns) > 2:
    print(f'  Effect Size Gradient:    r={r_g:+.4f}  {"✓ INCREASING WITH N" if r_g > 0.3 else "✗ FLAT"}')
    if slope != 0 and crossover_n > 0:
        print(f'  Crossover Point:         N ≈ {crossover_n:.0f} neurons')
print(f'\n  {len(df)}/900 COMPLETE. {remaining} remaining. GPU active.')
print('=' * 85)
