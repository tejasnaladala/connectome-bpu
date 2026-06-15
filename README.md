# Connectome Architecture Benchmark (CAB)

Do biological wiring diagrams make better neural network architectures than
random graphs? CAB answers that question with a controlled experiment: take a
real connectome, freeze it as the fixed weight matrix of a network, train only
a linear readout, and compare against four families of synthetic null graphs
matched in size and density.

The headline finding from the committed runs: on the *Drosophila* mushroom body,
the biological wiring beats the mean of its matched null graphs by **+7.4
accuracy points on the CartPole control task** (biological 0.134 vs null-mean
0.059), and by **+4.6 points averaged across all five non-trivial tasks**. The
mushroom body shows the largest organism-level advantage in the benchmark. It is
also the one circuit in biology built for high-fan-in associative memory, which
is the part of the result worth taking seriously.

That is the optimistic read. The honest read, which this repo reports in full,
is that the biological advantage is real but uneven. It is large for adult
*Drosophila* circuits on hard tasks, near zero on easy classification, and
negative for *C. elegans*. The point of the benchmark is to measure where the
advantage lives rather than to claim biology wins everywhere.

## What the committed results show

All numbers below come from `results/all_results.csv` (757 runs), excluding the
`Audio_Simulated` task, which saturates at 100% for every topology and carries no
signal. Reproduce them with `python scripts/quick_status.py` and
`python scripts/deep_analysis.py`.

**Pairwise win rate.** Of the 41 filled non-Audio organism-task cells, the
biological connectome beats the mean of its four matched nulls in **30, loses in
10, and ties in 1** (75% of decided pairs). A Wilcoxon signed-rank test on the 41
per-cell deltas gives p = 0.0009. The effect is small in absolute terms: the
overall biological-minus-null accuracy gap is +1.4 points, bootstrap 95% CI
[-6.0, +8.7] points, so the aggregate gap on its own is not distinguishable from
zero. The signal lives in the per-organism and per-task breakdown, not the grand
mean.

**Per-organism advantage** (biological mean minus null mean, accuracy points,
excluding Audio):

| Organism | Neurons | Δ accuracy |
|---|---:|---:|
| Adult mushroom body | 4,000 | +4.6 |
| Adult lateral horn | 4,000 | +3.3 |
| Adult optic lobe (medulla) | 4,000 | +2.9 |
| Adult central complex | 4,000 | +2.7 |
| Adult antennal lobe | 3,739 | +2.3 |
| Adult subesophageal zone | 4,000 | +1.8 |
| *Ciona intestinalis* | 177 | +0.9 |
| *Drosophila* larva | 2,952 | +0.7 |
| *C. elegans* male | 559 | -1.8 |
| *C. elegans* hermaphrodite | 419 | -2.1 |

Adult *Drosophila* brain regions help; the two *C. elegans* whole-animal
connectomes hurt. The *C. elegans* nervous system is a compact sensorimotor
circuit with positive degree assortativity, which is the opposite structural
profile from the disassortative, high-fan-in insect circuits that do well here.

**Per-task advantage.** The gap is largest on the hardest classification task and
disappears on the easiest:

| Task | Δ accuracy (bio − null) |
|---|---:|
| CIFAR-10 | +7.9 to +9.2 across adult circuits |
| FashionMNIST | +2.1 (mushroom body) |
| MNIST | +0.9 (mushroom body) |
| CartPole (RL) | high variance, +7.4 to -7.9 by organism |

CartPole appears at both extremes: the mushroom body gains +7.4 points while
*C. elegans* male loses 7.9. Reinforcement learning is where topology matters
most and also where the variance is highest, so treat the single-cell CartPole
numbers as suggestive rather than settled.

**Training speed.** The wall-clock story is task-dependent, not a flat win.
Biological networks train about **32% faster than the null mean on MNIST**
(411s vs 605s) and faster on the mushroom-body CIFAR-10 cell, but slower on
FashionMNIST and CartPole. Across all non-Audio runs the aggregate is roughly
6% *slower* for biological networks, dominated by the FashionMNIST and CartPole
cells; the per-cell median is close to even (+1%). There is no single speedup
number that summarizes this honestly, so the repo reports it per task.

## The degree-distribution result

The most useful null is the degree-preserved graph: it keeps the exact in- and
out-degree sequence of the biological connectome (Maslov-Sneppen edge swaps) and
randomizes everything else. If biology beats it, the advantage comes from
structure beyond the degree sequence.

In the committed data it does not, on average. The degree-preserved null reaches
the highest mean accuracy of any topology type (0.610 vs 0.595 biological,
excluding Audio), and on the mushroom body it edges out the real wiring on
CartPole. The straightforward reading is that for these tasks most of the
computational value of biological wiring is carried by its degree distribution,
the pattern of hub neurons and connection heterogeneity, rather than by motifs,
clustering, or community structure. This is the kind of result a benchmark is for:
it constrains the story rather than confirming a prior.

## Method

Following the Biological Processing Unit framework (Yu et al. 2025,
arXiv:2507.10951):

1. Input is linearly projected into the connectome dimension `N`.
2. The connectome adjacency `A` is used as a fixed `N x N` weight matrix. Edge
   weights are frozen random values; the topology is the only thing that varies
   between conditions. Sequential tasks use a recurrent variant where `A` defines
   the hidden-to-hidden connectivity.
3. A trainable linear readout maps the final hidden state to the task output. It
   is the only component that learns.

By freezing the internal weights, the benchmark isolates one variable: the
pattern of connections.

**Connectomes (10).** Four whole-animal or full-CNS reconstructions plus six
adult *Drosophila* brain regions. Counts are after removing isolated nodes; adult
regions are subsampled to the 4,000 highest-degree neurons.

| Connectome | Region | Neurons | Edges | Source |
|---|---|---:|---:|---|
| *C. elegans* herm. | whole animal | 419 | 4,647 | White 1986; Cook 2019 |
| *C. elegans* male | whole animal | 559 | 5,246 | Cook 2019 |
| *Ciona intestinalis* | CNS | 177 | 6,618 | Ryan et al. 2016 |
| *Drosophila* larva | whole brain | 2,952 | 110,140 | Winding 2023 |
| Adult antennal lobe | olfaction | 3,739 | 140,612 | FlyWire |
| Adult central complex | navigation | 4,000 | 299,789 | FlyWire |
| Adult lateral horn | olfaction | 4,000 | 215,808 | FlyWire |
| Adult mushroom body | learning/memory | 4,000 | 250,876 | FlyWire |
| Adult optic lobe (medulla) | vision | 4,000 | 262,027 | FlyWire |
| Adult subesophageal zone | feeding | 4,000 | 346,591 | FlyWire |

Graph-theoretic metrics for every connectome (clustering, modularity,
assortativity, spectral gap, rich-club, and more) are in
`results/graph_metrics.csv`.

**Null models (4), generated per connectome at matched density** (see
`models/controls.py`):

- **Erdős-Rényi** uniform random graph with the same edge count.
- **Barabási-Albert** preferential-attachment graph (scale-free degree tail).
- **Watts-Strogatz** ring-lattice with rewiring (small-world, high clustering).
- **Degree-preserved** Maslov-Sneppen edge swaps that keep the exact biological
  degree sequence and randomize higher-order structure.

**Tasks (6).** MNIST, FashionMNIST, CIFAR-10 (image classification), Sequential
MNIST (temporal), CartPole (reinforcement learning), and a simulated Audio task
that is excluded from analysis because every topology solves it perfectly.

**Design and what is committed.** The full design is 10 connectomes x 6 tasks x
5 topology types x 3 seeds = 900 runs. This repository ships the 757 runs
completed so far, covering 51 of the 60 organism-task cells; the 9 missing cells
are adult *Drosophila* regions on the most expensive tasks. Statistics are
computed only over filled cells, and the numbers above will shift as the
remaining cells fill in. Seeds are 42, 43, 44. Optimizer is Adam at 1e-3.

## Repository layout

```
models/
  bpu.py          BPU and SequentialBPU modules (frozen-topology networks)
  controls.py     the four null-model generators
  baselines.py    MLP and small-transformer reference baselines
  test_models.py  fast self-contained test suite
benchmarks/       one module per task + the runner orchestration
analysis/         graph metrics, correlations, error analysis, figures
scripts/          quick_status.py and deep_analysis.py (read the results CSV)
data/
  processed/      4 whole-animal connectome adjacency matrices (.npz)
  subcircuits/    6 adult Drosophila region adjacency matrices (.npz)
  raw/            source connectome files + standardization scripts
results/          all_results.csv and derived analysis tables
run_experiments.py  the driver that produced results/all_results.csv
```

## Reproducing

Requires Python 3.11, PyTorch, NumPy, SciPy, pandas, scikit-learn, NetworkX,
and tqdm. The image datasets are not committed; torchvision downloads MNIST,
FashionMNIST, and CIFAR-10 automatically on first run.

```bash
# Fast sanity check on the models and null generators (CPU, seconds)
python models/test_models.py

# Inspect the committed results
python scripts/quick_status.py
python scripts/deep_analysis.py

# Reproduce a single cheap cell (CPU is fine for the small connectomes)
python run_experiments.py --task MNIST --organism ciona --device cpu

# Full sweep (use a GPU; this is the run that produces all_results.csv)
python run_experiments.py --device cuda
```

`run_experiments.py` saves incrementally and resumes from whatever is already in
`results/all_results.csv`, so a sweep can be stopped and restarted. The full
sweep is GPU-scale work and takes many hours; the small connectomes
(*C. elegans*, *Ciona*) run on CPU in minutes per task.

## References

- Yu et al. (2025). Biological Processing Units. arXiv:2507.10951.
- White et al. (1986). The structure of the nervous system of *C. elegans*.
  *Phil. Trans. R. Soc. B*.
- Cook et al. (2019). Whole-animal connectomes of both *C. elegans* sexes.
  *Nature*.
- Ryan et al. (2016). The CNS connectome of a *Ciona intestinalis* larva.
  *eLife*.
- Winding et al. (2023). The connectome of an insect brain. *Science*.
- Dorkenwald et al. (2024). Neuronal wiring diagram of an adult brain (FlyWire).
  *Nature*.

## License

No license file is included yet. Until one is added, the connectome source data
remains under the terms of its original publications.
