#!/usr/bin/env python3
"""
BPU Connectome Experiment Dashboard -- Gamified TUI
====================================================
Live terminal dashboard with XP system, achievements, organism leaderboard,
bio-vs-random battle tracker, levels, and streaks.

Usage: python -m src.tui.dashboard
       python src/tui/dashboard.py
"""
import csv
import os
import subprocess
import sys
import time
import math
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# Force UTF-8 on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# -- Project root ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_CSV = ROOT / "results" / "all_results.csv"
METRICS_CSV = ROOT / "results" / "graph_metrics.csv"

# -- Constants -------------------------------------------------------------
ORGANISMS = [
    ("ciona", 177, "[cyan]CI[/]", "Tunicate"),
    ("celegans_herm", 419, "[green]CH[/]", "Nematode F"),
    ("celegans_male", 559, "[green]CM[/]", "Nematode M"),
    ("drosophila_larva", 2952, "[yellow]DL[/]", "Fruit Fly L"),
    ("adult_optic_lobe_medulla", 4000, "[magenta]OM[/]", "Optic Lobe"),
    ("adult_mushroom_body", 4000, "[magenta]MB[/]", "Mushroom B"),
    ("adult_central_complex", 4000, "[magenta]CX[/]", "Central Cx"),
    ("adult_antennal_lobe", 3739, "[magenta]AL[/]", "Antennal L"),
    ("adult_subesophageal_zone", 4000, "[magenta]SZ[/]", "Subesoph Z"),
    ("adult_lateral_horn", 4000, "[magenta]LH[/]", "Lateral H"),
]

TASKS = ["MNIST", "FashionMNIST", "CIFAR10", "SequentialMNIST", "Audio", "CartPole"]
CONTROLS = ["erdos_renyi", "barabasi_albert", "watts_strogatz", "degree_preserved"]
SEEDS = [42, 43, 44]

TOTAL_EXPERIMENTS = len(ORGANISMS) * (1 + len(CONTROLS)) * len(SEEDS) * len(TASKS)

# -- XP & Level System -----------------------------------------------------
XP_PER_EXPERIMENT = 100
XP_PER_HIGH_ACC = 50       # bonus for acc > 0.95
XP_PER_ORGANISM_DONE = 500  # all experiments for one organism
XP_PER_TASK_DONE = 300     # all experiments for one task
XP_PER_METRIC = 200        # graph metric computed

LEVEL_THRESHOLDS = [
    (0, "Neuron Novice", "[dim]"),
    (500, "Synapse Scout", "[bright_blue]"),
    (1500, "Dendrite Developer", "[cyan]"),
    (3000, "Axon Architect", "[green]"),
    (6000, "Cortex Commander", "[yellow]"),
    (10000, "Lobe Lord", "[bright_yellow]"),
    (18000, "Brain Baron", "[bright_magenta]"),
    (30000, "Connectome Champion", "[bright_red]"),
    (50000, "Neural Nexus", "[bold bright_white]"),
    (80000, "BPU Grandmaster", "[bold white on blue]"),
]

# -- Achievements -----------------------------------------------------------
ACHIEVEMENTS = {
    "first_blood":     ("First Blood",       "Complete 1 experiment",           1),
    "ten_strong":      ("Ten Strong",        "Complete 10 experiments",         10),
    "century":         ("Century Club",      "Complete 100 experiments",       100),
    "half_marathon":   ("Half Marathon",     "Complete 450 experiments",       450),
    "full_battery":    ("Full Battery",      "Complete all 900 experiments",   900),
    "sharpshooter":    ("Sharpshooter",      "Achieve >98% accuracy",          -1),
    "bio_wins":        ("Biology Wins",      "Bio beats ALL 4 controls",       -2),
    "species_complete":("Species Complete",  "Finish all tasks for 1 org",     -3),
    "metrics_master":  ("Metrics Master",    "All 10 graph metrics done",      -4),
    "speed_demon":     ("Speed Demon",       "Experiment under 60 seconds",    -5),
    "gpu_blazing":     ("GPU Blazing",       "GPU util above 90%",             -6),
    "small_world":     ("Small World",       "Find SW sigma > 10",             -7),
}

# -- Animation frames (ASCII-safe) -----------------------------------------
NEURON_FRAMES = [" o--o ", " o---o", "o----o", " o---o"]
BRAIN_FRAMES = [
    [" o-O-o ", " O-o-O ", " o-O-o "],
    [" O-o-O ", " o-O-o ", " O-o-O "],
    [" o-o-O ", " O-O-o ", " o-o-O "],
]
GPU_FRAMES = [
    "_..:||:.._..:||:..",
    ".:||:.._..::||:._.",
    ":||:.._..::||:._.:.",
    "||:.._..::||:._.:||",
    "|:.._..::||:._.:||:",
    ":.._..::||:._.:||:.",
    ".._..::||:._.:||:..",
    "._..::||:._.:||:.._",
]
BATTLE_FRAMES = [
    " [BIO] >>---> [RNG] ",
    " [BIO] >>>--> [RNG] ",
    " [BIO] >>>>-> [RNG] ",
    " [BIO] >>>>>  [RNG] ",
]

# -- Data loaders -----------------------------------------------------------
def load_results():
    results = []
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, "r") as f:
            for row in csv.DictReader(f):
                results.append(row)
    return results

def load_metrics():
    metrics = []
    if METRICS_CSV.exists():
        with open(METRICS_CSV, "r") as f:
            for row in csv.DictReader(f):
                metrics.append(row)
    return metrics

def get_gpu_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            return {
                "util": int(parts[0].strip()),
                "mem_used": int(parts[1].strip()),
                "mem_total": int(parts[2].strip()),
                "temp": int(parts[3].strip()),
                "power": float(parts[4].strip()),
            }
    except Exception:
        pass
    return None

def get_gpu_processes():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return len([l for l in result.stdout.strip().split("\n") if l.strip()])
    except Exception:
        pass
    return 0

# -- Gamification Engine ----------------------------------------------------
def compute_xp(results, metrics):
    xp = len(results) * XP_PER_EXPERIMENT
    xp += len(metrics) * XP_PER_METRIC

    # High accuracy bonuses
    for r in results:
        try:
            if float(r.get("accuracy", 0)) > 0.95:
                xp += XP_PER_HIGH_ACC
        except (ValueError, TypeError):
            pass

    # Organism completion bonus
    for org_name, _, _, _ in ORGANISMS:
        org_exps = [r for r in results if r.get("organism") == org_name or
                    r.get("name", "").startswith(org_name)]
        expected = (1 + len(CONTROLS)) * len(SEEDS) * len(TASKS)
        if len(org_exps) >= expected:
            xp += XP_PER_ORGANISM_DONE

    # Task completion bonus
    for task in TASKS:
        task_exps = [r for r in results if r.get("task") == task]
        expected = len(ORGANISMS) * (1 + len(CONTROLS)) * len(SEEDS)
        if len(task_exps) >= expected:
            xp += XP_PER_TASK_DONE

    return xp

def get_level(xp):
    level_name = LEVEL_THRESHOLDS[0][1]
    level_style = LEVEL_THRESHOLDS[0][2]
    level_num = 1
    next_threshold = LEVEL_THRESHOLDS[1][0] if len(LEVEL_THRESHOLDS) > 1 else xp + 1
    for i, (threshold, name, style) in enumerate(LEVEL_THRESHOLDS):
        if xp >= threshold:
            level_name = name
            level_style = style
            level_num = i + 1
            next_threshold = LEVEL_THRESHOLDS[i + 1][0] if i + 1 < len(LEVEL_THRESHOLDS) else threshold + 10000
    return level_num, level_name, level_style, next_threshold

def check_achievements(results, metrics, gpu_info):
    unlocked = []
    n = len(results)

    if n >= 1: unlocked.append("first_blood")
    if n >= 10: unlocked.append("ten_strong")
    if n >= 100: unlocked.append("century")
    if n >= 450: unlocked.append("half_marathon")
    if n >= 900: unlocked.append("full_battery")

    # Sharpshooter
    for r in results:
        try:
            if float(r.get("accuracy", 0)) > 0.98:
                unlocked.append("sharpshooter")
                break
        except (ValueError, TypeError):
            pass

    # Bio wins - check if bio > all 4 controls for any organism
    for org_name, _, _, _ in ORGANISMS:
        bio_accs = [float(r["accuracy"]) for r in results
                    if r.get("name") == org_name and r.get("type") == "biological"]
        if not bio_accs:
            continue
        bio_mean = sum(bio_accs) / len(bio_accs)
        all_ctrl_beaten = True
        ctrl_count = 0
        for ctrl in CONTROLS:
            ctrl_accs = [float(r["accuracy"]) for r in results
                         if r.get("name") == f"{org_name}_{ctrl}"]
            if ctrl_accs:
                ctrl_count += 1
                if sum(ctrl_accs)/len(ctrl_accs) >= bio_mean:
                    all_ctrl_beaten = False
        if all_ctrl_beaten and ctrl_count == 4:
            unlocked.append("bio_wins")
            break

    # Species complete
    for org_name, _, _, _ in ORGANISMS:
        org_exps = [r for r in results if r.get("organism") == org_name or
                    r.get("name", "").startswith(org_name)]
        expected = (1 + len(CONTROLS)) * len(SEEDS) * len(TASKS)
        if len(org_exps) >= expected:
            unlocked.append("species_complete")
            break

    # Metrics master
    if len(metrics) >= 10:
        unlocked.append("metrics_master")

    # Speed demon
    for r in results:
        try:
            if float(r.get("train_time_sec", 9999)) < 60:
                unlocked.append("speed_demon")
                break
        except (ValueError, TypeError):
            pass

    # GPU blazing
    if gpu_info and gpu_info.get("util", 0) > 90:
        unlocked.append("gpu_blazing")

    # Small world
    for m in metrics:
        try:
            if float(m.get("small_world_sigma", 0)) > 10:
                unlocked.append("small_world")
                break
        except (ValueError, TypeError):
            pass

    return list(set(unlocked))


# -- Panel Builders ---------------------------------------------------------
def build_header(frame, xp, level_num, level_name, level_style, next_xp):
    gpu_wave = GPU_FRAMES[frame % len(GPU_FRAMES)]
    xp_to_next = next_xp - xp
    xp_in_level = xp - (LEVEL_THRESHOLDS[level_num - 1][0] if level_num > 0 else 0)
    xp_level_range = next_xp - (LEVEL_THRESHOLDS[level_num - 1][0] if level_num > 0 else 0)
    xp_pct = min(xp_in_level / max(xp_level_range, 1), 1.0)
    xp_bar_w = 20
    xp_filled = int(xp_pct * xp_bar_w)

    t = Text()
    t.append("+=================================================================+\n", style="bold cyan")
    t.append("|  ", style="bold cyan")
    t.append(" BIOLOGICAL PROCESSING UNIT ", style="bold white on blue")
    t.append("  ", style="bold cyan")
    t.append(f"LVL {level_num} ", style="bold bright_yellow")
    t.append(f"{level_style}{level_name}[/]")
    t.append("  |\n", style="bold cyan")
    t.append("|  ", style="bold cyan")
    t.append(f"XP: {xp:,}  ", style="bold bright_yellow")
    t.append(f"[bright_yellow]{'#' * xp_filled}[/][dim]{'.' * (xp_bar_w - xp_filled)}[/]")
    t.append(f"  {xp_to_next:,} to next", style="dim")
    t.append("  |\n", style="bold cyan")
    t.append(f"|  GPU: ", style="bold cyan")
    t.append(f"{gpu_wave}", style="bold green")
    t.append(f"  {datetime.now().strftime('%H:%M:%S')}", style="dim")
    t.append("  |\n".rjust(24), style="bold cyan")
    t.append("+=================================================================+", style="bold cyan")

    return Panel(t, box=box.SIMPLE, style="bold cyan")


def build_organism_panel(results, frame):
    table = Table(box=box.SIMPLE_HEAVY, expand=True, show_header=True,
                  header_style="bold bright_white", padding=(0, 1))
    table.add_column("", width=3, justify="center")
    table.add_column("Organism", style="bold", width=16)
    table.add_column("N", justify="right", width=6, style="cyan")
    table.add_column("Progress", justify="center", width=14)
    table.add_column("Best", justify="right", width=8)
    table.add_column("Rank", justify="center", width=6)

    # Compute leaderboard ranking by best bio accuracy
    org_best = {}
    for org_name, n, emoji, label in ORGANISMS:
        bio = [r for r in results if r.get("name") == org_name and r.get("type") == "biological"]
        if bio:
            org_best[org_name] = max(float(r["accuracy"]) for r in bio)
    ranked = sorted(org_best.items(), key=lambda x: -x[1])
    rank_map = {name: i+1 for i, (name, _) in enumerate(ranked)}

    for org_name, n_neurons, emoji, label in ORGANISMS:
        all_org = [r for r in results if r.get("organism") == org_name or
                   r.get("name", "").startswith(org_name)]
        expected = (1 + len(CONTROLS)) * len(SEEDS) * len(TASKS)
        done = len(all_org)

        best_acc = ""
        if org_name in org_best:
            best_acc = f"[bold green]{org_best[org_name]:.4f}[/]"

        # Progress bar
        pct = done / max(expected, 1)
        bar_len = 8
        filled = int(pct * bar_len)
        if done == 0:
            bar = f"[dim]{'.' * bar_len}[/] 0"
        elif done < expected:
            neuron = NEURON_FRAMES[frame % len(NEURON_FRAMES)]
            bar = f"[yellow]{'#' * filled}[/][dim]{'.' * (bar_len - filled)}[/] {done}"
        else:
            bar = f"[bold green]{'#' * bar_len}[/] {done}"

        # Rank medal
        rank = rank_map.get(org_name, "-")
        if rank == 1:
            rank_str = "[bold bright_yellow]#1[/]"
        elif rank == 2:
            rank_str = "[bright_white]#2[/]"
        elif rank == 3:
            rank_str = "[yellow]#3[/]"
        elif isinstance(rank, int):
            rank_str = f"[dim]#{rank}[/]"
        else:
            rank_str = "[dim]-[/]"

        table.add_row(emoji, label, str(n_neurons), bar, best_acc, rank_str)

    return Panel(table, title="[bold bright_white]ORGANISM LEADERBOARD[/]",
                 border_style="green", box=box.ROUNDED)


def build_gpu_panel(frame):
    gpu = get_gpu_info()
    procs = get_gpu_processes()

    if gpu is None:
        return Panel("[dim]GPU info unavailable[/]", title="[bold]GPU[/]", border_style="red")

    t = Text()
    util = gpu["util"]
    bar_w = 28
    filled = int(util / 100 * bar_w)
    bar_c = "green" if util < 60 else "yellow" if util < 85 else "bold red"
    t.append(f"  GPU:  [{bar_c}]{'#' * filled}[/][dim]{'.' * (bar_w - filled)}[/] {util}%\n")

    mem_pct = gpu["mem_used"] / max(gpu["mem_total"], 1) * 100
    filled = int(mem_pct / 100 * bar_w)
    mc = "green" if mem_pct < 60 else "yellow" if mem_pct < 85 else "bold red"
    t.append(f"  VRAM: [{mc}]{'#' * filled}[/][dim]{'.' * (bar_w - filled)}[/] {gpu['mem_used']}/{gpu['mem_total']}MB\n")

    tc = "green" if gpu["temp"] < 70 else "yellow" if gpu["temp"] < 85 else "bold red"
    t.append(f"  Temp: [{tc}]{gpu['temp']}C[/]  Power: {gpu['power']:.0f}W  Procs: [cyan]{procs}[/]\n")
    wave = GPU_FRAMES[frame % len(GPU_FRAMES)]
    t.append(f"  [bold cyan]{wave}[/]")

    return Panel(t, title="[bold]>> RTX 5070 Ti <<[/]", border_style="bright_cyan", box=box.ROUNDED)


def build_results_panel(results):
    table = Table(box=box.SIMPLE, expand=True, show_header=True,
                  header_style="bold", padding=(0, 1))
    table.add_column("Network", width=20, style="cyan")
    table.add_column("Task", width=10)
    table.add_column("Type", width=14)
    table.add_column("Acc", justify="right", width=8)
    table.add_column("XP", justify="right", width=5)

    for r in results[-10:]:
        try:
            acc = float(r["accuracy"])
        except (ValueError, KeyError):
            acc = 0
        acc_s = "bold green" if acc > 0.95 else "yellow" if acc > 0.85 else "red"
        type_s = "bold bright_white" if r.get("type") == "biological" else "dim"
        short = r.get("name", "?").replace("adult_", "~").replace("celegans_", "ce_")
        xp_earned = XP_PER_EXPERIMENT + (XP_PER_HIGH_ACC if acc > 0.95 else 0)
        table.add_row(
            short, r.get("task", "?"),
            Text(r.get("type", "?"), style=type_s),
            Text(f"{acc:.4f}", style=acc_s),
            f"[bright_yellow]+{xp_earned}[/]"
        )

    return Panel(table, title=f"[bold]RESULTS LOG ({len(results)} total)[/]",
                 border_style="bright_yellow", box=box.ROUNDED)


def build_achievements_panel(unlocked, frame):
    t = Text()
    t.append("  ACHIEVEMENTS UNLOCKED\n\n", style="bold bright_yellow")

    for key, (name, desc, _) in ACHIEVEMENTS.items():
        if key in unlocked:
            sparkle = "*" if frame % 2 == 0 else "+"
            t.append(f"  [{sparkle}] ", style="bold bright_yellow")
            t.append(f"{name}", style="bold green")
            t.append(f" - {desc}\n", style="dim")
        else:
            t.append(f"  [ ] ", style="dim")
            t.append(f"{name}", style="dim")
            t.append(f" - {desc}\n", style="dim")

    progress = len(unlocked)
    total = len(ACHIEVEMENTS)
    t.append(f"\n  {progress}/{total} unlocked", style="bold bright_yellow")

    return Panel(t, title=f"[bold bright_yellow]TROPHIES {progress}/{total}[/]",
                 border_style="bright_yellow", box=box.ROUNDED)


def build_battle_panel(results, frame):
    """Bio vs Random — head-to-head battle tracker per organism."""
    t = Text()
    battle = BATTLE_FRAMES[frame % len(BATTLE_FRAMES)]
    t.append(f"  {battle}\n\n", style="bold bright_cyan")

    bio_wins_total = 0
    ctrl_wins_total = 0

    for org_name, _, emoji, label in ORGANISMS:
        bio_accs = [float(r["accuracy"]) for r in results
                    if r.get("name") == org_name and r.get("type") == "biological"]
        if not bio_accs:
            continue

        bio_mean = sum(bio_accs) / len(bio_accs)
        ctrl_all = []
        for ctrl in CONTROLS:
            ca = [float(r["accuracy"]) for r in results
                  if r.get("name") == f"{org_name}_{ctrl}"]
            ctrl_all.extend(ca)

        if not ctrl_all:
            t.append(f"  {emoji} {label:<12} bio={bio_mean:.4f} vs [dim]...[/]\n")
            continue

        ctrl_mean = sum(ctrl_all) / len(ctrl_all)
        diff = (bio_mean - ctrl_mean) * 100
        if diff > 0:
            bio_wins_total += 1
            color = "bold green"
            arrow = ">>"
        else:
            ctrl_wins_total += 1
            color = "bold red"
            arrow = "<<"

        bar_len = min(abs(int(diff * 10)), 10)
        bar = "#" * bar_len
        sign = "+" if diff > 0 else ""
        t.append(f"  {emoji} {label:<12} [{color}]{arrow} {sign}{diff:.2f}% {bar}[/]\n")

    if bio_wins_total + ctrl_wins_total > 0:
        t.append(f"\n  Score: [bold green]BIO {bio_wins_total}[/] - [bold red]{ctrl_wins_total} RANDOM[/]")
    else:
        t.append("  [dim]Waiting for matchups...[/]")

    return Panel(t, title="[bold]BATTLE: Bio vs Random[/]", border_style="bright_red", box=box.ROUNDED)


def build_scaling_panel(results, frame):
    bio_mnist = {}
    for r in results:
        if r.get("type") == "biological" and r.get("task") == "MNIST":
            name = r["name"]
            acc = float(r["accuracy"])
            if name not in bio_mnist or acc > bio_mnist[name]:
                bio_mnist[name] = acc

    if not bio_mnist:
        return Panel("[dim]Waiting for biological results...[/]",
                     title="[bold]H1: SCALING LAW[/]", border_style="magenta")

    org_neurons = {name: n for name, n, _, _ in ORGANISMS}
    t = Text()
    t.append("  Accuracy vs log(Neurons) -- MNIST\n\n", style="bold")

    chart_h, chart_w = 6, 35
    sorted_orgs = sorted(bio_mnist.items(), key=lambda x: org_neurons.get(x[0], 0))

    min_acc = min(bio_mnist.values()) - 0.01
    max_acc = max(bio_mnist.values()) + 0.01
    acc_range = max(max_acc - min_acc, 0.001)

    for row in range(chart_h, -1, -1):
        acc_val = min_acc + (row / chart_h) * acc_range
        line = f"  {acc_val:.3f} |"
        for i, (name, acc) in enumerate(sorted_orgs):
            col = int((i / max(len(sorted_orgs) - 1, 1)) * (chart_w - 2)) + 1
            row_pos = int((acc - min_acc) / acc_range * chart_h)
            if row_pos == row:
                sym = "O" if frame % 2 == 0 else "o"
                pad = col - (len(line) - 11)
                if pad > 0:
                    line += " " * pad + f"[bold green]{sym}[/]"
        t.append(line + "\n")

    t.append("         +" + "-" * chart_w + "\n")
    t.append("          ")
    for name, _ in sorted_orgs:
        t.append(f" {name[:4]}  ", style="dim")

    return Panel(t, title="[bold]H1: SCALING LAW[/]", border_style="magenta", box=box.ROUNDED)


def build_metrics_panel(metrics):
    if not metrics:
        return Panel("[dim]Computing graph metrics...[/]",
                     title="[bold]GRAPH METRICS[/]", border_style="blue")

    table = Table(box=box.SIMPLE, expand=True, show_header=True,
                  header_style="bold", padding=(0, 1))
    table.add_column("Network", width=18, style="cyan")
    table.add_column("N", justify="right", width=5)
    table.add_column("Clust", justify="right", width=6)
    table.add_column("SW", justify="right", width=6)
    table.add_column("Mod", justify="right", width=6)
    table.add_column("Recip", justify="right", width=6)

    for m in metrics:
        name = m.get("name", "?").replace("adult_", "~").replace("celegans_", "ce_")
        nodes = m.get("n_neurons", "?")
        clust = float(m.get("clustering_coeff", 0))
        sw = m.get("small_world_sigma", "0")
        modul = float(m.get("modularity", 0))
        recip = float(m.get("reciprocity", 0))

        try:
            sw_f = float(sw)
            sw_str = f"{sw_f:.1f}"
            sw_style = "bold green" if sw_f > 5 else "white"
        except ValueError:
            sw_str = "?"
            sw_style = "dim"

        table.add_row(name, str(nodes), f"{clust:.3f}",
                      Text(sw_str, style=sw_style), f"{modul:.3f}", f"{recip:.3f}")

    return Panel(table, title=f"[bold]GRAPH METRICS ({len(metrics)}/10)[/]",
                 border_style="blue", box=box.ROUNDED)


def build_pipeline_panel(results, metrics, frame):
    t = Text()
    task_counts = defaultdict(int)
    for r in results:
        task_counts[r.get("task", "")] += 1

    total_per_task = len(ORGANISMS) * (1 + len(CONTROLS)) * len(SEEDS)

    for task in TASKS:
        done = task_counts.get(task, 0)
        pct = done / total_per_task
        bar_len = 20
        filled = int(pct * bar_len)

        if done == 0:
            sym, color = "[.]", "dim"
        elif done < total_per_task:
            sym = ["[*]", "[+]", "[>]", "[#]"][frame % 4]
            color = "yellow"
        else:
            sym, color = "[v]", "green"

        bar = f"[{color}]{'#' * filled}[/][dim]{'.' * (bar_len - filled)}[/]"
        t.append(f"  {sym} {task:<16} {bar} {done}/{total_per_task}\n")

    overall_pct = len(results) / max(TOTAL_EXPERIMENTS, 1) * 100
    t.append(f"\n  Metrics: [cyan]{len(metrics)}/10[/]  ")
    t.append(f"Total: [bold]{len(results)}/{TOTAL_EXPERIMENTS}[/]  ")
    t.append(f"[bold bright_yellow]{overall_pct:.1f}%[/]")

    return Panel(t, title="[bold]PIPELINE PROGRESS[/]",
                 border_style="bright_white", box=box.DOUBLE)


def build_activity_log(results, frame):
    lines = []

    if results:
        last = results[-1]
        try:
            acc = float(last.get("accuracy", 0))
        except (ValueError, TypeError):
            acc = 0
        lines.append(f"  [green]OK[/] {last.get('name','?')} / {last.get('task','?')} -> {acc:.4f}")

    msgs = [
        "Training BPU on connectome...", "Forward pass through synapses...",
        "Backprop through projections...", "Evaluating test accuracy...",
        "Saving results to CSV...", "Loading adjacency matrix...",
        "Generating control network...", "Computing graph metrics...",
    ]
    spinner = ["|", "/", "-", "\\"][frame % 4]
    lines.append(f"  [yellow]{spinner}[/] {msgs[frame % len(msgs)]}")

    elapsed = timedelta(seconds=int(time.time()) % 86400)
    lines.append(f"  [dim]Uptime: {elapsed}[/]")

    return Panel("\n".join(lines), title="[bold]ACTIVITY[/]",
                 border_style="bright_blue", box=box.ROUNDED)


# -- Layout -----------------------------------------------------------------
def make_layout():
    layout = Layout(name="root")
    layout.split_column(
        Layout(name="header", size=6),
        Layout(name="body"),
        Layout(name="footer", size=12),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=3),
        Layout(name="right", ratio=2),
    )
    layout["left"].split_column(
        Layout(name="organisms", ratio=3),
        Layout(name="results", ratio=2),
    )
    layout["right"].split_column(
        Layout(name="gpu", size=7),
        Layout(name="battle", ratio=2),
        Layout(name="scaling", ratio=2),
    )
    layout["footer"].split_row(
        Layout(name="pipeline", ratio=2),
        Layout(name="achievements", ratio=2),
        Layout(name="metrics_activity", ratio=2),
    )
    layout["metrics_activity"].split_column(
        Layout(name="metrics", ratio=2),
        Layout(name="activity", size=5),
    )
    return layout


def main():
    console = Console()
    layout = make_layout()
    frame = 0
    prev_count = 0

    console.print("\n[bold bright_cyan]  BPU Connectome Dashboard starting...[/]\n")
    console.print("  [dim]Press Ctrl+C to exit[/]\n")

    with Live(layout, console=console, refresh_per_second=2, screen=True) as live:
        while True:
            try:
                results = load_results()
                metrics = load_metrics()
                gpu_info = get_gpu_info()

                # Gamification
                xp = compute_xp(results, metrics)
                level_num, level_name, level_style, next_xp = get_level(xp)
                unlocked = check_achievements(results, metrics, gpu_info)

                layout["header"].update(build_header(frame, xp, level_num, level_name, level_style, next_xp))
                layout["organisms"].update(build_organism_panel(results, frame))
                layout["results"].update(build_results_panel(results))
                layout["gpu"].update(build_gpu_panel(frame))
                layout["battle"].update(build_battle_panel(results, frame))
                layout["scaling"].update(build_scaling_panel(results, frame))
                layout["pipeline"].update(build_pipeline_panel(results, metrics, frame))
                layout["achievements"].update(build_achievements_panel(unlocked, frame))
                layout["metrics"].update(build_metrics_panel(metrics))
                layout["activity"].update(build_activity_log(results, frame))

                prev_count = len(results)
                frame += 1
                time.sleep(0.5)

            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/]")
                time.sleep(1)

    console.print("\n[bold green]  Dashboard stopped. Experiments continue in background.[/]\n")


if __name__ == "__main__":
    main()
