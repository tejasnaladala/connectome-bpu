"""Audio classification benchmark for BPU models.

Trains a BPU on speech command recognition using MFCC features.
Uses torchaudio SpeechCommands dataset (v0.02) with 35 command classes.
Falls back to a simulated audio benchmark if torchaudio is unavailable.
"""

import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.bpu import BPU


class SimulatedAudioDataset(Dataset):
    """Simulated audio classification dataset as fallback.

    Generates random MFCC-like features with fixed seeds for
    reproducibility. 35 classes, 400-dim features.
    """

    def __init__(self, n_samples=10000, d_features=400, n_classes=35, seed=0):
        rng = np.random.RandomState(seed)
        self.data = []
        self.labels = []
        per_class = n_samples // n_classes
        for c in range(n_classes):
            center = rng.randn(d_features).astype(np.float32) * 2.0
            samples = center + rng.randn(per_class, d_features).astype(np.float32) * 0.5
            self.data.append(samples)
            self.labels.extend([c] * per_class)
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.array(self.labels, dtype=np.int64)
        mean = self.data.mean(axis=0)
        std = self.data.std(axis=0) + 1e-8
        self.data = (self.data - mean) / std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])


def _build_speech_commands_loaders(data_dir, batch_size, seed):
    """Build train/test loaders from torchaudio SpeechCommands.

    Returns (train_loader, test_loader, d_features, n_classes) or
    raises ImportError/RuntimeError on failure.
    """
    import torchaudio
    from torchaudio.datasets import SPEECHCOMMANDS
    from torchaudio.transforms import MFCC

    class SpeechCommandsSubset(SPEECHCOMMANDS):
        def __init__(self, root, subset):
            super().__init__(root, download=True, subset=subset)

    train_set = SpeechCommandsSubset(data_dir, subset="training")
    test_set = SpeechCommandsSubset(data_dir, subset="testing")

    all_labels = sorted(set(item[2] for item in train_set))
    label_to_idx = {label: i for i, label in enumerate(all_labels)}
    n_classes = len(label_to_idx)

    n_mfcc = 40
    target_length = 16000  # 1 second at 16kHz
    mfcc_transform = MFCC(sample_rate=16000, n_mfcc=n_mfcc)

    class MFCCDataset(Dataset):
        """Wraps SpeechCommands to produce averaged MFCC features."""

        def __init__(self, speech_dataset, label_map, mfcc_fn, target_len):
            self.dataset = speech_dataset
            self.label_map = label_map
            self.mfcc_fn = mfcc_fn
            self.target_len = target_len

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            waveform, sample_rate, label, *_ = self.dataset[idx]
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            if waveform.size(1) < self.target_len:
                padding = self.target_len - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            else:
                waveform = waveform[:, :self.target_len]
            mfcc = self.mfcc_fn(waveform)  # (1, n_mfcc, T)
            features = mfcc.squeeze(0).mean(dim=1)  # (n_mfcc,)
            label_idx = self.label_map[label]
            return features, label_idx

    train_mfcc = MFCCDataset(train_set, label_to_idx, mfcc_transform, target_length)
    test_mfcc = MFCCDataset(test_set, label_to_idx, mfcc_transform, target_length)

    train_loader = DataLoader(train_mfcc, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_mfcc, batch_size=batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, test_loader, n_mfcc, n_classes


def run_audio(adjacency, name="unnamed", epochs=20, lr=1e-3, batch_size=64,
              device="cuda", seed=42):
    """Train a BPU on speech command classification.

    Uses torchaudio SpeechCommands (v0.02) with MFCC features.
    Falls back to simulated audio benchmark if torchaudio is unavailable.

    Args:
        adjacency: (N, N) numpy array or scipy sparse matrix.
        name: Human-readable name for this connectome/configuration.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.
        batch_size: Mini-batch size.
        device: Torch device string.
        seed: Random seed for reproducibility.

    Returns:
        dict with keys: name, task, n_neurons, accuracy, final_loss,
        train_time_sec, learnable_params, fixed_params, epochs, lr, seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "speech_commands")

    try:
        train_loader, test_loader, d_features, n_classes = \
            _build_speech_commands_loaders(data_dir, batch_size, seed)
        task_name = "Audio_SpeechCommands"
    except (ImportError, RuntimeError, OSError) as e:
        print(f"[audio] torchaudio SpeechCommands unavailable ({e}), "
              f"using simulated audio benchmark.")
        d_features = 400
        n_classes = 35
        task_name = "Audio_Simulated"

        full_dataset = SimulatedAudioDataset(
            n_samples=10500, d_features=d_features,
            n_classes=n_classes, seed=seed
        )
        n_train = int(0.85 * len(full_dataset))
        n_test = len(full_dataset) - n_train
        train_ds, test_ds = random_split(
            full_dataset, [n_train, n_test],
            generator=torch.Generator().manual_seed(seed)
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size,
                                 shuffle=False, num_workers=0, pin_memory=True)

    # --- Model ---
    model = BPU(adjacency, d_in=d_features, d_out=n_classes).to(device)
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
        final_loss = epoch_loss / max(n_batches, 1)

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

    accuracy = correct / max(total, 1)

    return {
        "name": name,
        "task": task_name,
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
