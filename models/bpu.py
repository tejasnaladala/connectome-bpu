"""Biological Processing Unit (BPU) models.

Core neural network modules that use fixed biological connectome wiring
as the recurrent weight matrix, with learnable input/output projections.
"""

import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


def _sparse_to_dense_tensor(adj):
    """Convert a scipy sparse matrix or numpy array to a dense torch tensor."""
    if sp.issparse(adj):
        adj = adj.toarray()
    return torch.tensor(adj, dtype=torch.float32)


ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
}


class BPU(nn.Module):
    """Biological Processing Unit for feedforward tasks.

    Architecture:
        Input -> Linear(D_in, N) -> act -> Fixed_Linear(N, N) -> act -> Linear(N, D_out)

    The recurrent (hidden-to-hidden) layer uses the *transpose* of the
    biological adjacency matrix as its fixed, non-learnable weight.
    Input and output projections are standard learnable linear layers.

    Args:
        adjacency: (N, N) scipy sparse matrix or numpy ndarray. Biological
            connectome adjacency matrix.
        d_in: Input feature dimension.
        d_out: Output dimension.
        n_recurrent_steps: Number of times to apply the fixed recurrent
            layer (message-passing steps). Default 1.
        activation: Activation function name. One of 'relu', 'tanh',
            'sigmoid', 'gelu'. Default 'relu'.
    """

    def __init__(self, adjacency, d_in, d_out, n_recurrent_steps=1, activation="relu"):
        super().__init__()
        self.N = adjacency.shape[0]
        assert adjacency.shape == (self.N, self.N), (
            f"Adjacency must be square, got {adjacency.shape}"
        )
        self.d_in = d_in
        self.d_out = d_out
        self.n_recurrent_steps = n_recurrent_steps

        # Learnable projections
        self.input_proj = nn.Linear(d_in, self.N)
        self.output_proj = nn.Linear(self.N, d_out)

        # Fixed recurrent weight: transpose of adjacency
        W_rec = _sparse_to_dense_tensor(adjacency).T
        self.register_buffer("W_rec", W_rec)

        # Bias for recurrent layer (learnable)
        self.recurrent_bias = nn.Parameter(torch.zeros(self.N))

        # Activation
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(ACTIVATIONS)}")
        self.act = ACTIVATIONS[activation]()

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, d_in) input tensor.

        Returns:
            (batch, d_out) output tensor.
        """
        # Input projection
        h = self.act(self.input_proj(x))  # (batch, N)

        # Fixed recurrent steps
        for _ in range(self.n_recurrent_steps):
            h = self.act(torch.mm(h, self.W_rec.T) + self.recurrent_bias)  # (batch, N)

        # Output projection
        return self.output_proj(h)  # (batch, d_out)

    def count_learnable_params(self):
        """Count the number of learnable (requires_grad=True) parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_fixed_params(self):
        """Count the number of fixed (non-learnable) parameters in buffers."""
        return sum(b.numel() for b in self.buffers())

    def __repr__(self):
        return (
            f"BPU(N={self.N}, d_in={self.d_in}, d_out={self.d_out}, "
            f"recurrent_steps={self.n_recurrent_steps}, "
            f"learnable={self.count_learnable_params():,}, "
            f"fixed={self.count_fixed_params():,})"
        )


class SequentialBPU(nn.Module):
    """Biological Processing Unit for temporal/sequential tasks.

    Processes input one timestep at a time while maintaining a hidden
    state h that is propagated through the fixed biological wiring.

    At each timestep t:
        inp = act(input_proj(x_t))
        h   = act(W_rec @ (h + inp) + bias)

    After all timesteps, the final hidden state is projected to the output.

    Args:
        adjacency: (N, N) scipy sparse matrix or numpy ndarray.
        d_in_per_step: Input dimension at each timestep.
        d_out: Output dimension.
        activation: Activation function name. Default 'relu'.
    """

    def __init__(self, adjacency, d_in_per_step, d_out, activation="relu"):
        super().__init__()
        self.N = adjacency.shape[0]
        assert adjacency.shape == (self.N, self.N), (
            f"Adjacency must be square, got {adjacency.shape}"
        )
        self.d_in_per_step = d_in_per_step
        self.d_out = d_out

        # Learnable projections
        self.input_proj = nn.Linear(d_in_per_step, self.N)
        self.output_proj = nn.Linear(self.N, d_out)

        # Fixed recurrent weight: transpose of adjacency
        W_rec = _sparse_to_dense_tensor(adjacency).T
        self.register_buffer("W_rec", W_rec)

        # Bias for recurrent layer (learnable)
        self.recurrent_bias = nn.Parameter(torch.zeros(self.N))

        # Activation
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(ACTIVATIONS)}")
        self.act = ACTIVATIONS[activation]()

    def forward(self, x_seq):
        """Forward pass over a sequence.

        Args:
            x_seq: (batch, seq_len, d_in_per_step) input tensor.

        Returns:
            (batch, d_out) output tensor from the final hidden state.
        """
        batch_size, seq_len, _ = x_seq.shape
        device = x_seq.device

        # Initialize hidden state to zeros
        h = torch.zeros(batch_size, self.N, device=device, dtype=x_seq.dtype)

        for t in range(seq_len):
            x_t = x_seq[:, t, :]  # (batch, d_in_per_step)
            inp = self.act(self.input_proj(x_t))  # (batch, N)
            h = self.act(
                torch.mm(h + inp, self.W_rec.T) + self.recurrent_bias
            )  # (batch, N)

        return self.output_proj(h)  # (batch, d_out)

    def count_learnable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_fixed_params(self):
        return sum(b.numel() for b in self.buffers())

    def __repr__(self):
        return (
            f"SequentialBPU(N={self.N}, d_in_per_step={self.d_in_per_step}, "
            f"d_out={self.d_out}, "
            f"learnable={self.count_learnable_params():,}, "
            f"fixed={self.count_fixed_params():,})"
        )
