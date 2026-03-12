"""Fashion-MNIST benchmark for BPU models.

Trains a BPU on the Fashion-MNIST classification task.
Input: 784 (28x28 flattened), Output: 10 clothing categories.
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
from models.bpu import BPU


def run_fashion_mnist(adjacency, name="unnamed", epochs=20, lr=1e-3,
                      batch_size=128, device="cuda", seed=42):
    """Train a BPU on Fashion-MNIST classification.

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
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "fashion_mnist")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)),  # flatten to 784
    ])

    train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                          transform=transform)
    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                         transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    # --- Model ---
    model = BPU(adjacency, d_in=784, d_out=10).to(device)
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
            optimizer.zero_grad()
            output = model(data)
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
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    accuracy = correct / total

    return {
        "name": name,
        "task": "FashionMNIST",
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
