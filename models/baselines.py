"""Baseline models for comparison with BPU architectures.

These provide standard neural network baselines (MLP, Transformer)
matched in parameter count to enable fair comparison with biologically
wired BPU models.
"""

import math
import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """Standard multi-layer perceptron baseline.

    A fully connected feedforward network with configurable depth and
    width. Use hidden_size and n_layers to roughly match the total
    parameter count of a corresponding BPU model.

    Args:
        d_in: Input dimension.
        d_out: Output dimension.
        hidden_size: Width of each hidden layer.
        n_layers: Number of hidden layers. Default 2.
    """

    def __init__(self, d_in, d_out, hidden_size, n_layers=2):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        layers = []
        in_dim = d_in
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, d_out))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, d_in) input tensor.

        Returns:
            (batch, d_out) output tensor.
        """
        return self.net(x)

    def count_params(self):
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class SmallTransformer(nn.Module):
    """Small transformer baseline for sequence or token-level tasks.

    A compact transformer encoder followed by mean pooling and a
    linear output head. Designed to operate within the same parameter
    budget as a BPU model for fair comparison.

    For non-sequential inputs, the input is treated as a single token.

    Args:
        d_in: Input feature dimension (per token).
        d_out: Output dimension.
        d_model: Transformer hidden dimension. Default 64.
        n_heads: Number of attention heads. Default 4.
        n_layers: Number of transformer encoder layers. Default 2.
        max_seq_len: Maximum sequence length for positional encoding.
            Default 1024.
    """

    def __init__(self, d_in, d_out, d_model=64, n_heads=4, n_layers=2,
                 max_seq_len=1024):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(d_in, d_model)

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_out),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Either (batch, d_in) for single-token input, or
               (batch, seq_len, d_in) for sequential input.

        Returns:
            (batch, d_out) output tensor.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, d_in)

        batch_size, seq_len, _ = x.shape

        # Project to model dimension and add positional encoding
        h = self.input_proj(x)  # (batch, seq_len, d_model)
        h = h + self.pos_embedding[:, :seq_len, :]

        # Transformer encoding
        h = self.encoder(h)  # (batch, seq_len, d_model)

        # Mean pool over sequence dimension
        h = h.mean(dim=1)  # (batch, d_model)

        return self.output_head(h)  # (batch, d_out)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
