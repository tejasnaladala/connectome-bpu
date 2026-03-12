"""Benchmark runners for BPU models on standard ML tasks."""

from benchmarks.mnist import run_mnist
from benchmarks.fashion_mnist import run_fashion_mnist
from benchmarks.cifar10 import run_cifar10
from benchmarks.sequential_mnist import run_sequential_mnist
from benchmarks.audio import run_audio
from benchmarks.cartpole import run_cartpole
