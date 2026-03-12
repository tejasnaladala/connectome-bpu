"""Sequential MNIST benchmark for BPU models.

Trains a SequentialBPU on pixel-by-pixel MNIST classification.
Each of the 784 pixels is fed one at a time (d_in_per_step=1).
The model must classify the digit from its final hidden state.
Tests temporal processing and memory capacity.
"""

import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.bpu import SequentialBPU


def run_sequential_mnist(adjacency, name="unnamed", epochs=20, lr=1e-3,
                         batch_size=128, device="cuda", seed=42):
    """Train a SequentialBPU on pixel-by-pixel MNIST.

    Pixels are fed one at a time (784 timesteps, d_in_per_step=1).
    Classification is based on the final hidden state.

    Args:
        adjacency: (N, N) numpy array or scipy sparse matrix.
        name: Human-readable name for this connectome/configuration.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.
        batch_size: Mini-batch size.
        device: Torch device string ('cuda' or 'cpu').
        seed: Random seed for reproducibility.

    Returns:
        dict with keys: name, task, n_neurons, accuracy, final_loss,
        train_time_sec, learnable_params, fixed_params, epochs, lr, seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # --- Data ---
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "mnist")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                  transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    # --- Model ---
    # 784 timesteps, 1 pixel per step, 10 output classes
    model = SequentialBPU(adjacency, d_in_per_step=1, d_out=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Train ---
    t0 = time.time()
    final_loss = 0.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            # Reshape: (batch, 1, 28, 28) -> (batch, 784, 1)
            x_seq = data.view(data.size(0), -1, 1)
            optimizer.zero_grad()
            output = model(x_seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        final_loss = epoch_loss / n_batches

    train_time = time.time() - t0

    # --- Test ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            x_seq = data.view(data.size(0), -1, 1)
            output = model(x_seq)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    accuracy = correct / total

    return {
        "name": name,
        "task": "SequentialMNIST",
        "n_neurons": model.N,
        "accuracy": accuracy,
        "final_loss": final_loss,
        "train_time_sec": round(train_time, 2),
        "learnable_params": model.count_learnable_params(),
        "fixed_params": model.count_fixed_params(),
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
    }
