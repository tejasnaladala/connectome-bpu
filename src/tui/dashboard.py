#!/usr/bin/env python3
"""
BPU Connectome Experiment Dashboard -- Retro Terminal Edition
=============================================================
CRT-style phosphor terminal with live experiment tracking,
GPU telemetry, hypothesis testing, and statistical analysis.

Usage: python src/tui/dashboard.py
"""
import csv, os, subprocess, sys, time, math, random
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_CSV = ROOT / "results" / "all_results.csv"
METRICS_CSV = ROOT / "results" / "graph_metrics.csv"

ORGANISMS = [
    ("ciona", 177, "Ciona intestinalis"),
    ("celegans_herm", 419, "C. elegans (herm)"),
    ("celegans_male", 559, "C. elegans (male)"),
    ("drosophila_larva", 2952, "Drosophila larva"),
    ("adult_optic_lobe_medulla", 4000, "Optic Medulla"),
    ("adult_mushroom_body", 4000, "Mushroom Body"),
    ("adult_central_complex", 4000, "Central Complex"),
    ("adult_antennal_lobe", 3739, "Antennal Lobe"),
    ("adult_subesophageal_zone", 4000, "Subesoph. Zone"),
    ("adult_lateral_horn", 4000, "Lateral Horn"),
]
TASKS = ["MNIST", "FashionMNIST", "CIFAR10", "SequentialMNIST", "Audio", "CartPole"]
CONTROLS = ["erdos_renyi", "barabasi_albert", "watts_strogatz", "degree_preserved"]
SEEDS = [42, 43, 44]
NETS_PER_ORG = 1 + len(CONTROLS)
EXPS_PER_TASK = len(ORGANISMS) * NETS_PER_ORG * len(SEEDS)
TOTAL = len(ORGANISMS) * NETS_PER_ORG * len(SEEDS) * len(TASKS)

# -- Retro Terminal Animations --
SCAN_FRAMES = [
    ">>>------->",
    "->>>------>",
    "-->>>----->",
    "--->>>---->",
    "---->>>--->",
    "----->>>-->",
    "------>>>->",
    "------->>>>",
]
CURSOR_FRAMES = ["_", " "]
SIGNAL_FRAMES = [
    "[----|----]",
    "[--+-|----]",
    "[----|--+-]",
    "[--+-|--+-]",
]

# -- Retro color palette: amber CRT + green phosphor --
AMBER = "bold yellow"
DIM_AMBER = "yellow"
GREEN = "bold green"
DIM_GREEN = "green"
PHOSPHOR = "bright_green"
COLD = "bright_cyan"
WARN = "bright_red"
DIM = "dim"

def load_csv(path):
    rows = []
    if path.exists():
        with open(path, "r") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows

def gpu_info():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.gr,clocks.mem,fan.speed",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            p = r.stdout.strip().split(",")
            return {"util": int(p[0]), "mem_used": int(p[1]), "mem_total": int(p[2]),
                    "temp": int(p[3]), "power": float(p[4]),
                    "clk_gpu": p[5].strip(), "clk_mem": p[6].strip(), "fan": p[7].strip()}
    except Exception:
        pass
    return None

def gpu_procs():
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            lines = [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]
            return [{"pid": l.split(",")[0].strip(), "mem": l.split(",")[1].strip() if "," in l else "?"} for l in lines]
    except Exception:
        pass
    return []

def safe_float(val, default=0.0):
    try: return float(val)
    except (ValueError, TypeError): return default

def cohen_d(g1, g2):
    if len(g1) < 1 or len(g2) < 1: return 0.0
    m1, m2 = statistics.mean(g1), statistics.mean(g2)
    if len(g1) < 2 and len(g2) < 2: return 0.0
    s1 = statistics.stdev(g1) if len(g1) > 1 else 0.001
    s2 = statistics.stdev(g2) if len(g2) > 1 else 0.001
    ps = math.sqrt((s1**2 + s2**2) / 2)
    return (m1 - m2) / ps if ps > 0 else 0.0

def pearson_r(xs, ys):
    if len(xs) < 3: return 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = math.sqrt(sum((x - mx)**2 for x in xs))
    sy = math.sqrt(sum((y - my)**2 for y in ys))
    if sx == 0 or sy == 0: return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


# -- Panels ----------------------------------------------------------------

def header_panel(results, frame):
    done = len(results)
    pct = done / max(TOTAL, 1) * 100
    times = [safe_float(r.get("train_time_sec")) for r in results if safe_float(r.get("train_time_sec")) > 0]
    avg_t = statistics.mean(times) if times else 0
    n_procs = max(len(gpu_procs()), 1)
    remaining = TOTAL - done
    eta_s = (remaining * avg_t) / n_procs if avg_t > 0 else 0
    eta_td = timedelta(seconds=int(eta_s)) if eta_s > 0 else None
    finish = (datetime.now() + timedelta(seconds=eta_s)).strftime("%b %d %H:%M") if eta_s > 0 else "---"

    scan = SCAN_FRAMES[frame % len(SCAN_FRAMES)]
    cursor = CURSOR_FRAMES[frame % len(CURSOR_FRAMES)]

    bar_w = 40
    filled = int(pct / 100 * bar_w)

    t = Text()
    t.append("  CONNECTOME ARCHITECTURE BENCHMARK ", style="bold bright_white on dark_green")
    t.append(f"  {scan}", style=DIM_GREEN)
    t.append(f"  {datetime.now().strftime('%H:%M:%S')}{cursor}\n", style=DIM)

    t.append(f" [{PHOSPHOR}]{'=' * filled}[/][{DIM}]{'-' * (bar_w - filled)}[/]")
    t.append(f" [{GREEN}]{done}[/]/{TOTAL} [{AMBER}]{pct:.1f}%[/]  ")
    t.append(f"ETA [{COLD}]{eta_td or '---'}[/]  ")
    t.append(f"DONE [{COLD}]{finish}[/]  ")
    t.append(f"~{avg_t:.0f}s/exp  {n_procs} proc", style=DIM)

    return Panel(t, border_style=DIM_GREEN, box=box.DOUBLE)

def gpu_panel(frame):
    g = gpu_info()
    procs = gpu_procs()
    signal = SIGNAL_FRAMES[frame % len(SIGNAL_FRAMES)]
    if not g:
        return Panel(f"[{DIM}]{signal} GPU OFFLINE {signal}[/]", title=f"[{WARN}]GPU[/]", border_style="red")

    t = Text()
    bar_w = 25
    for label, val, mx, unit in [("UTL", g["util"], 100, "%"), ("MEM", g["mem_used"], g["mem_total"], "MB")]:
        pct_v = val / max(mx, 1) * 100
        filled = int(pct_v / 100 * bar_w)
        c = PHOSPHOR if pct_v < 60 else DIM_AMBER if pct_v < 85 else WARN
        t.append(f" {label} [{c}]{'|' * filled}[/][{DIM}]{'.' * (bar_w - filled)}[/] {val}/{mx}{unit}\n")

    tc = PHOSPHOR if g["temp"] < 70 else DIM_AMBER if g["temp"] < 85 else WARN
    t.append(f" [{tc}]{g['temp']}C[/] {g['power']:.0f}W  [{DIM}]{g['clk_gpu']}MHz  Fan:{g['fan']}[/]")
    t.append(f" [{DIM_GREEN}]{signal}[/]")
    if procs:
        t.append(f"\n [{DIM_GREEN}]PID: {', '.join(p['pid'] for p in procs)}[/]")

    return Panel(t, title=f"[{GREEN}]RTX 5070 Ti[/]", border_style=DIM_GREEN, box=box.HEAVY)

def organism_panel(results, frame):
    table = Table(box=box.SIMPLE_HEAVY, expand=True, show_header=True,
                  header_style="bold bright_white", padding=(0, 1))
    table.add_column("Organism", width=18, style=DIM_GREEN)
    table.add_column("N", justify="right", width=5, style=DIM_AMBER)
    table.add_column("Done", justify="right", width=9)
    table.add_column("Progress", width=20)
    table.add_column("Bio", justify="right", width=7)
    table.add_column("Ctrl", justify="right", width=7)
    table.add_column("Delta", justify="right", width=8)
    table.add_column("", width=3)

    for i, (org_name, n, label) in enumerate(ORGANISMS):
        all_org = [r for r in results if r.get("organism") == org_name or r.get("name", "").startswith(org_name)]
        expected = NETS_PER_ORG * len(SEEDS) * len(TASKS)
        done = len(all_org)

        bio = [safe_float(r["accuracy"]) for r in all_org if r.get("type") == "biological"]
        ctrl = [safe_float(r["accuracy"]) for r in all_org if r.get("type") != "biological"]

        pct = done / max(expected, 1)
        bar_len = 12
        filled = int(pct * bar_len)
        if done == 0:
            bar = f"[{DIM}]{'.' * bar_len}[/]"
        elif done < expected:
            bar = f"[{DIM_GREEN}]{'=' * filled}[/][{DIM}]{'-' * (bar_len - filled)}[/]"
        else:
            bar = f"[{PHOSPHOR}]{'=' * bar_len}[/]"
        bar += f" {done}/{expected}"

        bio_str = f"{statistics.mean(bio):.3f}" if bio else f"[{DIM}]-[/]"
        ctrl_str = f"{statistics.mean(ctrl):.3f}" if ctrl else f"[{DIM}]-[/]"
        delta_str = ""
        if bio and ctrl:
            d = (statistics.mean(bio) - statistics.mean(ctrl)) * 100
            c = PHOSPHOR if d > 0 else WARN
            delta_str = f"[{c}]{'+' if d > 0 else ''}{d:.1f}%[/]"

        # Activity indicator
        indicator = ""
        if done > 0 and done < expected:
            indicator = f"[{DIM_GREEN}]>[/]"

        table.add_row(label, str(n), f"{done}", bar, bio_str, ctrl_str, delta_str, indicator)

    return Panel(table, title=f"[{GREEN}]ORGANISMS[/]", border_style=DIM_GREEN, box=box.HEAVY)

def scope_panel(frame):
    """Compact system scope readout replacing the old brain animation."""
    signal = SIGNAL_FRAMES[frame % len(SIGNAL_FRAMES)]
    scan = SCAN_FRAMES[frame % len(SCAN_FRAMES)]
    cursor = CURSOR_FRAMES[frame % len(CURSOR_FRAMES)]

    t = Text()
    t.append(f"  {signal} NEURAL TOPOLOGY SCAN\n", style=DIM_GREEN)
    t.append(f"  {scan}\n", style=DIM_AMBER)
    t.append(f"  10 organisms  |  6 tasks  |  5 topologies\n", style=DIM)
    t.append(f"  3 seeds/exp   |  900 total experiments\n", style=DIM)
    t.append(f"  BIO vs ER/BA/WS/DP controls{cursor}", style=DIM_GREEN)

    return Panel(t, title=f"[{DIM_AMBER}]SCOPE[/]", border_style=DIM_AMBER, box=box.ROUNDED)

def stats_panel(results):
    t = Text()
    if not results:
        t.append(f" [{DIM}]Awaiting data...[/]")
        return Panel(t, title=f"[{COLD}]STATS[/]", border_style=COLD)

    bio = [safe_float(r["accuracy"]) for r in results if r.get("type") == "biological"]
    ctrl = [safe_float(r["accuracy"]) for r in results if r.get("type") != "biological"]
    bio_loss = [safe_float(r["final_loss"]) for r in results if r.get("type") == "biological"]
    ctrl_loss = [safe_float(r["final_loss"]) for r in results if r.get("type") != "biological"]

    t.append(" ACCURACY\n", style=AMBER)
    if bio:
        t.append(f"  Bio:  n={len(bio):3d} u={statistics.mean(bio):.4f}")
        if len(bio) > 1: t.append(f" s={statistics.stdev(bio):.4f}")
        t.append("\n")
    if ctrl:
        t.append(f"  Ctrl: n={len(ctrl):3d} u={statistics.mean(ctrl):.4f}")
        if len(ctrl) > 1: t.append(f" s={statistics.stdev(ctrl):.4f}")
        t.append("\n")
    if bio and ctrl:
        diff = statistics.mean(bio) - statistics.mean(ctrl)
        c = PHOSPHOR if diff > 0 else WARN
        d = cohen_d(bio, ctrl)
        t.append(f"  [{c}]D={'+' if diff>0 else ''}{diff*100:.3f}%  d={d:+.3f}[/]\n")

    if bio_loss and ctrl_loss:
        t.append(" LOSS CONVERGENCE\n", style=DIM_AMBER)
        ld = cohen_d(ctrl_loss, bio_loss)
        lm = statistics.mean(bio_loss) - statistics.mean(ctrl_loss)
        lc = PHOSPHOR if lm < 0 else WARN
        t.append(f"  Bio:{statistics.mean(bio_loss):.4f} Ctrl:{statistics.mean(ctrl_loss):.4f}")
        t.append(f" [{lc}]d={ld:+.3f}{'  BIO BETTER' if ld > 0.2 else ''}[/]\n")

    # Pairwise wins
    wins, losses = 0, 0
    for r in results:
        if r.get("type") == "biological":
            org, task, seed = r.get("organism", ""), r.get("task", ""), r.get("seed", "")
            ba = safe_float(r["accuracy"])
            for cr in results:
                if cr.get("organism") == org and cr.get("task") == task and cr.get("seed") == seed and cr.get("type") != "biological":
                    if ba > safe_float(cr["accuracy"]): wins += 1
                    elif ba < safe_float(cr["accuracy"]): losses += 1
    tp = wins + losses
    if tp > 0:
        t.append(" PAIRWISE\n", style=COLD)
        wr = wins / tp * 100
        wc = PHOSPHOR if wr > 55 else DIM_AMBER if wr > 45 else WARN
        t.append(f"  Bio [{wc}]{wins}/{tp} ({wr:.1f}%)[/] Ctrl {losses}/{tp}\n")

    # Effect gradient
    efx = []
    for org_name, n, _ in ORGANISMS:
        ob = [safe_float(r["accuracy"]) for r in results if r.get("type") == "biological" and r.get("organism") == org_name]
        oc = [safe_float(r["accuracy"]) for r in results if r.get("type") != "biological" and r.get("organism") == org_name]
        if ob and oc:
            d = cohen_d(ob, oc)
            if abs(d) < 1e6: efx.append((n, d, org_name))
    if len(efx) >= 2:
        t.append(" EFFECT GRADIENT\n", style=DIM_GREEN)
        for n, d, name in sorted(efx):
            dc = PHOSPHOR if d > 0.2 else DIM_AMBER if d > -0.2 else WARN
            short = name.replace("adult_", "~").replace("celegans_", "ce_")
            bar_len = min(int(abs(d) * 5), 10)
            bar_c = "|" * bar_len
            t.append(f"  {n:>5}n [{dc}]{bar_c:<10} d={d:+.3f}[/] {short}\n")

    # Controls
    t.append(" CONTROLS\n", style="bright_white")
    for cn in CONTROLS:
        ca = [safe_float(r["accuracy"]) for r in results if r.get("type") == cn]
        if ca and bio:
            d = (statistics.mean(bio) - statistics.mean(ca)) * 100
            dc = PHOSPHOR if d > 0 else WARN
            t.append(f"  {cn:<16} [{dc}]{'+' if d>0 else ''}{d:.2f}%[/]\n")

    times = [safe_float(r.get("train_time_sec")) for r in results if safe_float(r.get("train_time_sec")) > 0]
    if times:
        t.append(f" [{AMBER}]TIME[/] avg={statistics.mean(times):.0f}s  total={sum(times)/3600:.1f}h\n")

    return Panel(t, title=f"[{COLD}]STATISTICS[/]", border_style=COLD, box=box.HEAVY)

def hypothesis_panel(results, metrics, frame):
    t = Text()
    org_n = {name: n for name, n, _ in ORGANISMS}

    # H1
    bio_by_org = defaultdict(list)
    for r in results:
        if r.get("type") == "biological":
            bio_by_org[r.get("organism", r["name"])].append(safe_float(r["accuracy"]))

    t.append(" H1 Scaling: ", style=AMBER)
    if len(bio_by_org) >= 2:
        pts = sorted([(org_n.get(n, 0), statistics.mean(a)) for n, a in bio_by_org.items() if n in org_n])
        if len(pts) >= 3:
            r_val = pearson_r([math.log(n) for n, _ in pts], [a for _, a in pts])
            rc = PHOSPHOR if r_val > 0.5 else DIM_AMBER if r_val > 0 else WARN
            t.append(f"r(logN,acc)=[{rc}]{r_val:+.3f}[/]  {len(pts)} orgs")
        else:
            t.append(f"{len(pts)} orgs")
    else:
        t.append(f"[{DIM}]{len(bio_by_org)}/10[/]")
    t.append("\n")

    # H2
    ba = [safe_float(r["accuracy"]) for r in results if r.get("type") == "biological"]
    ca = [safe_float(r["accuracy"]) for r in results if r.get("type") != "biological"]
    t.append(" H2 Bio>Rand: ", style=DIM_AMBER)
    if ba and ca:
        d = statistics.mean(ba) - statistics.mean(ca)
        cd = cohen_d(ba, ca)
        t.append(f"[{PHOSPHOR if d > 0 else WARN}]{'+' if d>0 else ''}{d*100:.2f}% d={cd:+.3f}[/]")
    else:
        t.append(f"[{DIM}]...[/]")
    t.append("\n")

    # Gradient
    efx = []
    for org_name, n, _ in ORGANISMS:
        ob = [safe_float(r["accuracy"]) for r in results if r.get("type") == "biological" and r.get("organism") == org_name]
        oc = [safe_float(r["accuracy"]) for r in results if r.get("type") != "biological" and r.get("organism") == org_name]
        if ob and oc:
            d = cohen_d(ob, oc)
            if abs(d) < 1e6: efx.append((n, d))
    t.append(" GRADIENT: ", style=COLD)
    if len(efx) >= 3:
        gr = pearson_r([math.log(n) for n, _ in sorted(efx)], [d for _, d in sorted(efx)])
        gc = PHOSPHOR if gr > 0.5 else DIM_AMBER if gr > 0 else WARN
        t.append(f"r(logN,d)=[{gc}]{gr:+.3f}[/]  ")
        for n, d in sorted(efx):
            if d > 0:
                t.append(f"crossover~{n}n")
                break
    else:
        t.append(f"[{DIM}]{len(efx)} orgs[/]")
    t.append("\n")

    # Loss
    bl = [safe_float(r["final_loss"]) for r in results if r.get("type") == "biological"]
    cl = [safe_float(r["final_loss"]) for r in results if r.get("type") != "biological"]
    t.append(" LOSS: ", style=DIM_GREEN)
    if bl and cl:
        ld = cohen_d(cl, bl)
        t.append(f"[{PHOSPHOR if ld > 0.2 else DIM_AMBER if ld > 0 else WARN}]d={ld:+.3f} {'BIO BETTER' if ld > 0.2 else ''}[/]\n")
    else:
        t.append(f"[{DIM}]...[/]\n")

    # H3-H5
    t.append(f" H3:{len(metrics)}/10  ", style="bright_white")
    sc = len(set(r.get("organism") for r in results if "adult_" in r.get("name", "")))
    t.append(f"H4:{sc}/6  H5:")
    herm = [safe_float(r["accuracy"]) for r in results if r.get("name") == "celegans_herm" and r.get("type") == "biological"]
    male = [safe_float(r["accuracy"]) for r in results if r.get("name") == "celegans_male" and r.get("type") == "biological"]
    if herm and male:
        d = (statistics.mean(male) - statistics.mean(herm)) * 100
        t.append(f"[{PHOSPHOR if d > 0 else DIM_AMBER}]{'+' if d>0 else ''}{d:.1f}%[/]")
    else:
        t.append(f"[{DIM}]--[/]")

    return Panel(t, title=f"[{DIM_AMBER}]HYPOTHESES[/]", border_style=DIM_AMBER, box=box.DOUBLE)

def task_panel(results, frame):
    table = Table(box=box.SIMPLE, expand=True, show_header=True,
                  header_style="bold bright_white", padding=(0, 1))
    table.add_column("Task", width=14, style=DIM_GREEN)
    table.add_column("Progress", width=22)
    table.add_column("Done", justify="right", width=7)
    table.add_column("Mean", justify="right", width=7)
    table.add_column("Best", justify="right", width=7)
    table.add_column("Avg t", justify="right", width=6)

    by_task = defaultdict(list)
    for r in results: by_task[r.get("task", "")].append(r)

    for i, task in enumerate(TASKS):
        rows = by_task.get(task, [])
        done = len(rows)
        total = EXPS_PER_TASK
        pct = done / max(total, 1)
        bar_len = 14
        filled = int(pct * bar_len)

        if done == 0:
            bar = f"[{DIM}]{'.' * bar_len}[/]"
        elif done < total:
            bar = f"[{DIM_GREEN}]{'=' * filled}[/][{DIM}]{'-' * (bar_len - filled)}[/]"
        else:
            bar = f"[{PHOSPHOR}]{'=' * bar_len}[/]"

        accs = [safe_float(r.get("accuracy")) for r in rows]
        times = [safe_float(r.get("train_time_sec")) for r in rows if safe_float(r.get("train_time_sec")) > 0]

        table.add_row(
            task, bar, f"{done}/{total}",
            f"{statistics.mean(accs):.3f}" if accs else "-",
            f"[{PHOSPHOR}]{max(accs):.3f}[/]" if accs else "-",
            f"{statistics.mean(times):.0f}s" if times else "-",
        )

    return Panel(table, title=f"[{AMBER}]TASKS[/]", border_style=DIM_AMBER, box=box.DOUBLE)

def results_panel(results, frame):
    table = Table(box=box.SIMPLE, expand=True, show_header=True,
                  header_style="bold bright_white", padding=(0, 1))
    for col, w in [("Network", 20), ("Task", 9), ("Type", 12), ("Acc", 7), ("Loss", 9), ("t", 5)]:
        table.add_column(col, width=w, justify="left" if col in ("Network","Task","Type") else "right")

    for r in results[-10:]:
        acc = safe_float(r.get("accuracy"))
        ac = PHOSPHOR if acc > 0.95 else DIM_AMBER if acc > 0.85 else WARN
        ts = GREEN if r.get("type") == "biological" else DIM
        name = r.get("name", "?").replace("adult_", "~").replace("celegans_", "ce_")
        table.add_row(
            name, r.get("task", "?"), Text(r.get("type", "?"), style=ts),
            Text(f"{acc:.4f}", style=ac),
            f"{safe_float(r.get('final_loss')):.5f}",
            f"{safe_float(r.get('train_time_sec')):.0f}s",
        )

    return Panel(table, title=f"[bright_white]LOG ({len(results)})[/]", border_style=DIM, box=box.ROUNDED)

def active_panel(results, frame):
    t = Text()
    cursor = CURSOR_FRAMES[frame % len(CURSOR_FRAMES)]
    if not results:
        t.append(f" [{DIM_GREEN}]>>{cursor}[/] Awaiting first result...")
        return Panel(t, title=f"[{DIM_AMBER}]ACTIVE[/]", border_style=DIM_AMBER)

    last = results[-1]
    try: elapsed = time.time() - os.path.getmtime(RESULTS_CSV)
    except OSError: elapsed = 0

    times = [safe_float(r.get("train_time_sec")) for r in results if safe_float(r.get("train_time_sec")) > 0]
    avg = statistics.mean(times) if times else 300
    exp_pct = min(elapsed / max(avg, 1) * 100, 99)
    bar_w = 25
    filled = int(exp_pct / 100 * bar_w)

    t.append(f" [{DIM_GREEN}]>>{cursor}[/] ")
    t.append(f"{last.get('organism', '?')}/{last.get('task', '?')} ")
    t.append(f"[{DIM_GREEN}]{'=' * filled}[/][{DIM}]{'-' * (bar_w - filled)}[/] {exp_pct:.0f}%")
    t.append(f"  {int(elapsed)}s/~{avg:.0f}s")

    return Panel(t, title=f"[{DIM_AMBER}]ACTIVE[/]", border_style=DIM_AMBER, box=box.ROUNDED)


def make_layout():
    layout = Layout(name="root")
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="body"),
        Layout(name="footer", size=12),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=3),
        Layout(name="right", ratio=2),
    )
    layout["left"].split_column(
        Layout(name="organisms"),
        Layout(name="active", size=3),
    )
    layout["right"].split_column(
        Layout(name="gpu", size=5),
        Layout(name="scope", size=8),
        Layout(name="stats"),
    )
    layout["footer"].split_row(
        Layout(name="tasks", ratio=2),
        Layout(name="hypotheses", ratio=2),
        Layout(name="results_log", ratio=2),
    )
    return layout


def main():
    console = Console()
    layout = make_layout()
    frame = 0

    with Live(layout, console=console, refresh_per_second=2, screen=True) as live:
        while True:
            try:
                results = load_csv(RESULTS_CSV)
                metrics = load_csv(METRICS_CSV)

                layout["header"].update(header_panel(results, frame))
                layout["organisms"].update(organism_panel(results, frame))
                layout["active"].update(active_panel(results, frame))
                layout["gpu"].update(gpu_panel(frame))
                layout["scope"].update(scope_panel(frame))
                layout["stats"].update(stats_panel(results))
                layout["hypotheses"].update(hypothesis_panel(results, metrics, frame))
                layout["tasks"].update(task_panel(results, frame))
                layout["results_log"].update(results_panel(results, frame))

                frame += 1
                time.sleep(0.5)
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[{WARN}]{e}[/]")
                time.sleep(1)

    console.print(f"\n[{PHOSPHOR}]Terminal closed. Experiments continue in background.[/]\n")

if __name__ == "__main__":
    main()
