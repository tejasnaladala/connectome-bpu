import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.sparse import save_npz

from models.bpu import BPU, SequentialBPU
from models.controls import generate_all_controls
from data.scripts.standardize import standardize_connectome
import run_experiments

load_all_connectomes = run_experiments.load_all_connectomes


def _adjacency(n: int = 64, edge_count: int = 320, seed: int = 7) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    possible = n * (n - 1)
    flat = rng.choice(possible, size=edge_count, replace=False)
    rows = flat // (n - 1)
    cols = flat % (n - 1)
    cols = cols + (cols >= rows)
    weights = rng.uniform(0.1, 1.0, size=edge_count)
    return sp.csr_matrix((weights, (rows, cols)), shape=(n, n))


def test_every_null_has_the_biological_edge_count() -> None:
    biological = _adjacency()

    controls = generate_all_controls(biological, seed=19)

    assert controls
    for name, adjacency in controls.items():
        assert adjacency.nnz == biological.nnz, (
            f"{name} has {adjacency.nnz} edges; expected {biological.nnz}"
        )


def test_degree_preserved_null_keeps_directed_degree_sequences() -> None:
    biological = _adjacency()

    shuffled = generate_all_controls(biological, seed=23)["degree_preserved"]

    np.testing.assert_array_equal(
        shuffled.getnnz(axis=0), biological.getnnz(axis=0)
    )
    np.testing.assert_array_equal(
        shuffled.getnnz(axis=1), biological.getnnz(axis=1)
    )


def test_every_null_keeps_the_biological_weight_distribution() -> None:
    biological = _adjacency()

    controls = generate_all_controls(biological, seed=29)

    expected = np.sort(biological.data)
    for name, adjacency in controls.items():
        np.testing.assert_allclose(
            np.sort(adjacency.data),
            expected,
            err_msg=f"{name} changed the recurrent weight distribution",
        )


def test_feedforward_protocol_trains_only_the_readout() -> None:
    model = BPU(_adjacency(n=24, edge_count=72), d_in=8, d_out=3)

    learnable = {name for name, value in model.named_parameters() if value.requires_grad}

    assert learnable == {"output_proj.weight", "output_proj.bias"}


def test_sequential_protocol_trains_only_the_readout() -> None:
    model = SequentialBPU(
        _adjacency(n=24, edge_count=72), d_in_per_step=2, d_out=3
    )

    learnable = {name for name, value in model.named_parameters() if value.requires_grad}

    assert learnable == {"output_proj.weight", "output_proj.bias"}


def _write_connectome(tmp_path, source_kind: str) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    adjacency_path = processed / "example_adjacency.npz"
    save_npz(adjacency_path, _adjacency(n=12, edge_count=30))
    digest = hashlib.sha256(adjacency_path.read_bytes()).hexdigest()
    metadata = {
        "name": "example",
        "provenance": {
            "source_kind": source_kind,
            "source_url": "https://example.org/connectome",
            "citation": "Example et al. (2026)",
            "artifact_sha256": digest,
        },
    }
    (processed / "example_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def test_loader_rejects_synthetic_connectomes(tmp_path) -> None:
    _write_connectome(tmp_path, source_kind="synthetic")

    with pytest.raises(ValueError, match="measured connectome"):
        load_all_connectomes(base_dir=tmp_path)


def test_loader_accepts_a_digest_verified_measured_connectome(tmp_path) -> None:
    _write_connectome(tmp_path, source_kind="measured")

    connectomes = load_all_connectomes(base_dir=tmp_path)

    assert list(connectomes) == ["example"]


def test_loader_rejects_a_changed_connectome_artifact(tmp_path) -> None:
    _write_connectome(tmp_path, source_kind="measured")
    adjacency_path = tmp_path / "data" / "processed" / "example_adjacency.npz"
    adjacency_path.write_bytes(adjacency_path.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="SHA-256"):
        load_all_connectomes(base_dir=tmp_path)


def test_standardization_writes_a_digest_bound_provenance_record(tmp_path) -> None:
    edge_frame = pd.DataFrame(
        {
            "source": ["a", "b", "c"],
            "target": ["b", "c", "a"],
            "weight": [2, 3, 1],
        }
    )
    provenance = {
        "source_kind": "measured",
        "source_url": "https://example.org/source",
        "citation": "Example et al. (2026)",
    }

    _, metadata = standardize_connectome(
        "example",
        edge_frame,
        "source",
        "target",
        "weight",
        output_dir=tmp_path,
        provenance=provenance,
    )

    artifact_path = tmp_path / "example_adjacency.npz"
    expected_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert metadata["provenance"] == {
        **provenance,
        "artifact_sha256": expected_digest,
    }


def test_invalidated_result_artifact_is_not_reported_as_a_finding() -> None:
    root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "quick_status.py")],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "invalidated" in (completed.stdout + completed.stderr).lower()


def test_corrected_protocol_writes_to_a_new_result_artifact() -> None:
    assert (
        run_experiments.PROTOCOL_VERSION
        == "cab-v2-readout-only-density-matched"
    )
    assert Path(run_experiments.RESULTS_CSV).name == "cab_v2_results.csv"


def test_each_new_result_carries_protocol_and_source_provenance() -> None:
    adjacency = _adjacency(n=12, edge_count=30)

    def task(**kwargs):
        return {"accuracy": 0.5}

    result = run_experiments.run_single_experiment(
        adjacency,
        "example",
        "example",
        "biological",
        "ExampleTask",
        {"func": task, "epochs": 1, "batch_size": 2},
        "cpu",
        42,
        source_artifact_sha256="a" * 64,
        biological_edge_count=adjacency.nnz,
    )

    assert result["protocol_version"] == run_experiments.PROTOCOL_VERSION
    assert result["source_artifact_sha256"] == "a" * 64
    assert result["graph_edge_count"] == adjacency.nnz
    assert result["biological_edge_count"] == adjacency.nnz


@pytest.mark.parametrize("legacy_runner", ["run_all.py", "benchmarks/runner.py"])
def test_legacy_runners_cannot_bypass_the_corrected_protocol(
    legacy_runner: str,
) -> None:
    root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, str(root / legacy_runner), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "run_experiments.py" in completed.stderr


def test_every_legacy_result_reader_checks_artifact_status() -> None:
    root = Path(__file__).resolve().parents[1]
    readers = [
        path
        for path in root.rglob("*.py")
        if "all_results.csv" in path.read_text(encoding="utf-8")
        and path != Path(__file__)
    ]

    assert readers
    for reader in readers:
        source = reader.read_text(encoding="utf-8")
        assert "require_reportable_artifact" in source, reader.relative_to(root)


def test_synthetic_ciona_builder_is_retired() -> None:
    root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, str(root / "data" / "scripts" / "build_ciona.py")],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "measured" in completed.stderr.lower()
