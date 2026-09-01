# Connectome Architecture Benchmark

An empirical test of a narrow question: after controlling for size, edge count,
and recurrent-weight distribution, does biological wiring provide a useful
inductive bias for small neural systems?

## Current status

**No biological advantage is currently claimed.**

An audit of the original experiment found five protocol failures:

- the input projection and recurrent bias learned even though the stated method
  said only the output readout was trained;
- the Barabasi-Albert and Watts-Strogatz nulls had substantially fewer edges
  than the biological graph;
- 90 Ciona rows used a synthetic stand-in rather than the measured connectome;
- result rows were not bound to immutable source-data fingerprints;
- CartPole reported the final training episodes rather than a separate
  evaluation rollout.

The original 757-row artifact remains in `results/all_results.csv` so the
failure is inspectable. `results/ARTIFACT_STATUS.json` marks it as invalidated,
and the analysis scripts refuse to report it as a finding unless an explicit
forensic override is supplied.

The corrected protocol is implemented and tested. It has **not** been run at
full scale, so there is no replacement benchmark result yet.

## Corrected protocol

CAB v2 uses the protocol identifier
`cab-v2-readout-only-density-matched`.

1. Every connectome must be measured data with a citation, source URL, and a
   SHA-256 digest of the standardized adjacency artifact.
2. The input projection, recurrent adjacency, and recurrent bias are fixed.
   Only the linear output readout is trainable.
3. Each null has the same node count and exact directed edge count as its
   biological counterpart.
4. Each null receives the biological graph's exact nonzero-weight multiset,
   shuffled over the null topology. The degree-preserved null also keeps the
   exact directed in-degree and out-degree sequences.
5. Every new result row records the protocol version, source-artifact digest,
   graph edge count, and biological edge count.
6. Corrected runs write to `results/cab_v2_results.csv`; they cannot resume into
   or overwrite the invalidated artifact.

The four null families are Erdos-Renyi, Barabasi-Albert, Watts-Strogatz, and a
directed degree-preserving edge shuffle.

## What is verified

The fast test suite checks:

- feedforward and sequential output shapes;
- fixed recurrent weights;
- readout-only optimization;
- exact edge-count matching for every null;
- exact weight-distribution matching for every null;
- directed degree preservation;
- rejection of synthetic, missing, or digest-mismatched connectomes;
- provenance propagation during standardization;
- isolation of the old result artifact;
- retirement of the two legacy runners that bypassed these controls.

Run it with:

```bash
python -m pytest -q
python models/test_models.py
```

## Running CAB v2

Connectome data are not distributed in this repository. Place each source edge
list under `data/raw/<name>/` with a `provenance.json` containing:

```json
{
  "source_kind": "measured",
  "source_url": "https://source.example/data",
  "citation": "Authors (year), title"
}
```

Then standardize and run a bounded cell:

```bash
python data/scripts/standardize.py
python run_experiments.py --task MNIST --organism <name> --device cpu
```

The full cross-species matrix is GPU-scale work. It should be run only after
the source manifests and a held-out analysis plan have been reviewed.

`run_all.py` and `benchmarks/runner.py` are intentionally retired. They could
bypass provenance checks and write incompatible results.

## Remaining limitations

- CAB v2 has not produced a complete immutable result artifact.
- Exact edge-count and weight-distribution matching do not control every graph
  statistic. Density, degree, reciprocity, motifs, path length, and community
  structure require separate null hypotheses.
- A fixed random input projection is still one modeling choice; conclusions may
  depend on its seed and dimensionality.
- The task set is a convenience sample, not evidence of general computational
  superiority or biological mechanism.
- CartPole is excluded from CAB v2 until it has a separately seeded,
  update-free evaluation protocol.
- Multiple organisms, tasks, nulls, and seeds require a predeclared statistical
  analysis with multiplicity control.

## Repository map

```text
models/                 fixed-connectome modules and null generators
benchmarks/             task implementations
data/scripts/           provenance-aware standardization
tests/test_protocol.py  protocol regressions
results/                invalidated historical artifact and status record
run_experiments.py      the only supported experiment runner
```

## References and data terms

The benchmark design was motivated by the Biological Processing Unit framework
(Yu et al., 2025, arXiv:2507.10951). Candidate connectome sources and their data
terms are listed in `DATA_LICENSE.md`. Source data retain their original
licenses; the MIT license in this repository applies only to authored code and
documentation.
