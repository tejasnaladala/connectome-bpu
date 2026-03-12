"""Tests for BPU models, controls, and baselines.

Run with: python models/test_models.py
"""

import sys
import traceback
import numpy as np
import scipy.sparse as sp
import torch

from bpu import BPU, SequentialBPU
from controls import (
    erdos_renyi,
    barabasi_albert,
    watts_strogatz,
    degree_preserved_shuffle,
    generate_all_controls,
)
from baselines import MLPBaseline, SmallTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_adjacency(N=50, density=0.1, seed=0):
    """Create a random sparse adjacency matrix for testing."""
    rng = np.random.RandomState(seed)
    mask = rng.rand(N, N) < density
    np.fill_diagonal(mask, 0)
    weights = rng.rand(N, N) * mask
    return sp.csr_matrix(weights)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_bpu_forward():
    """BPU output shape is (batch, d_out)."""
    N, d_in, d_out, batch = 50, 10, 5, 8
    adj = make_test_adjacency(N)
    model = BPU(adj, d_in, d_out)
    x = torch.randn(batch, d_in)
    out = model(x)
    assert out.shape == (batch, d_out), (
        f"Expected ({batch}, {d_out}), got {out.shape}"
    )
    # Test with multiple recurrent steps
    model2 = BPU(adj, d_in, d_out, n_recurrent_steps=3)
    out2 = model2(x)
    assert out2.shape == (batch, d_out), (
        f"Multi-step: expected ({batch}, {d_out}), got {out2.shape}"
    )
    print("  PASS test_bpu_forward")


def test_bpu_fixed_weights():
    """Recurrent weights (W_rec buffer) must have requires_grad=False."""
    adj = make_test_adjacency(50)
    model = BPU(adj, 10, 5)

    # W_rec is a buffer, not a parameter => requires_grad is False
    assert hasattr(model, "W_rec"), "Model missing W_rec buffer"
    assert not model.W_rec.requires_grad, "W_rec should not require grad"

    # Verify W_rec is the transpose of the adjacency
    adj_dense = torch.tensor(adj.toarray(), dtype=torch.float32)
    assert torch.allclose(model.W_rec, adj_dense.T, atol=1e-6), (
        "W_rec should be the transpose of the adjacency matrix"
    )
    print("  PASS test_bpu_fixed_weights")


def test_bpu_learnable_projections():
    """BPU must have at least 2 learnable parameter groups (input + output proj)."""
    adj = make_test_adjacency(50)
    model = BPU(adj, 10, 5)
    learnable_groups = [
        name for name, p in model.named_parameters() if p.requires_grad
    ]
    assert len(learnable_groups) >= 2, (
        f"Expected >= 2 learnable param groups, got {len(learnable_groups)}: "
        f"{learnable_groups}"
    )
    # Verify input_proj and output_proj weights are present
    param_names = set(learnable_groups)
    assert "input_proj.weight" in param_names, "Missing input_proj.weight"
    assert "output_proj.weight" in param_names, "Missing output_proj.weight"
    print("  PASS test_bpu_learnable_projections")


def test_sequential_bpu_forward():
    """SequentialBPU processes (batch, seq_len, d_in_per_step) -> (batch, d_out)."""
    N, d_in_step, d_out = 50, 4, 5
    batch, seq_len = 8, 20
    adj = make_test_adjacency(N)
    model = SequentialBPU(adj, d_in_step, d_out)
    x = torch.randn(batch, seq_len, d_in_step)
    out = model(x)
    assert out.shape == (batch, d_out), (
        f"Expected ({batch}, {d_out}), got {out.shape}"
    )

    # Different sequence lengths should work
    x_short = torch.randn(batch, 5, d_in_step)
    out_short = model(x_short)
    assert out_short.shape == (batch, d_out), (
        f"Short seq: expected ({batch}, {d_out}), got {out_short.shape}"
    )
    print("  PASS test_sequential_bpu_forward")


def test_controls_match_size():
    """All control graphs must have the same N as the biological adjacency."""
    N = 50
    bio_adj = make_test_adjacency(N, density=0.15)
    controls = generate_all_controls(bio_adj, seed=42)

    for name, ctrl_adj in controls.items():
        assert ctrl_adj.shape == (N, N), (
            f"Control '{name}' has shape {ctrl_adj.shape}, expected ({N}, {N})"
        )
        # Verify weights are in [0, 1]
        if ctrl_adj.nnz > 0:
            assert ctrl_adj.data.min() >= -1e-9, (
                f"Control '{name}' has negative weights"
            )
            assert ctrl_adj.data.max() <= 1.0 + 1e-9, (
                f"Control '{name}' has weights > 1"
            )
    print("  PASS test_controls_match_size")


def test_controls_different_wiring():
    """Control graphs should have different edge patterns than the biological graph."""
    N = 60
    bio_adj = make_test_adjacency(N, density=0.15, seed=99)
    controls = generate_all_controls(bio_adj, seed=42)

    bio_dense = bio_adj.toarray()
    bio_nonzero = set(zip(*bio_adj.nonzero()))

    for name, ctrl_adj in controls.items():
        ctrl_nonzero = set(zip(*ctrl_adj.nonzero()))
        # Edge sets should not be identical (there can be overlap, but not 100%)
        if len(bio_nonzero) > 0 and len(ctrl_nonzero) > 0:
            overlap = len(bio_nonzero & ctrl_nonzero)
            total = len(bio_nonzero | ctrl_nonzero)
            jaccard = overlap / total if total > 0 else 0
            assert jaccard < 0.99, (
                f"Control '{name}' has Jaccard similarity {jaccard:.3f} with "
                f"biological graph (too similar)"
            )
    print("  PASS test_controls_different_wiring")


def test_mlp_baseline():
    """MLP baseline forward pass works correctly."""
    d_in, d_out, hidden, batch = 10, 5, 64, 8
    model = MLPBaseline(d_in, d_out, hidden, n_layers=2)
    x = torch.randn(batch, d_in)
    out = model(x)
    assert out.shape == (batch, d_out), (
        f"Expected ({batch}, {d_out}), got {out.shape}"
    )
    assert model.count_params() > 0, "MLP should have parameters"

    # Test with different layer counts
    model3 = MLPBaseline(d_in, d_out, hidden, n_layers=3)
    out3 = model3(x)
    assert out3.shape == (batch, d_out)
    assert model3.count_params() > model.count_params(), (
        "3-layer MLP should have more params than 2-layer"
    )
    print("  PASS test_mlp_baseline")


def test_transformer_baseline():
    """SmallTransformer forward pass works for both 2D and 3D inputs."""
    d_in, d_out, batch = 10, 5, 8
    model = SmallTransformer(d_in, d_out, d_model=32, n_heads=4, n_layers=1)

    # 2D input (single token)
    x_2d = torch.randn(batch, d_in)
    out_2d = model(x_2d)
    assert out_2d.shape == (batch, d_out), (
        f"2D: expected ({batch}, {d_out}), got {out_2d.shape}"
    )

    # 3D input (sequence)
    seq_len = 16
    x_3d = torch.randn(batch, seq_len, d_in)
    out_3d = model(x_3d)
    assert out_3d.shape == (batch, d_out), (
        f"3D: expected ({batch}, {d_out}), got {out_3d.shape}"
    )
    print("  PASS test_transformer_baseline")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_bpu_forward,
    test_bpu_fixed_weights,
    test_bpu_learnable_projections,
    test_sequential_bpu_forward,
    test_controls_match_size,
    test_controls_different_wiring,
    test_mlp_baseline,
    test_transformer_baseline,
]


def main():
    print(f"Running {len(ALL_TESTS)} tests...\n")
    passed = 0
    failed = 0
    errors = []

    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, e))
            print(f"  FAIL {test_fn.__name__}: {e}")
            traceback.print_exc()
            print()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)}")
    print(f"{'='*60}")

    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
