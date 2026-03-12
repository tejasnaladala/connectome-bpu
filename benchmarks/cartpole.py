"""CartPole reinforcement learning benchmark for BPU models.

Trains a BPU as a policy network using REINFORCE on CartPole-v1.
Input: 4 (cart position, velocity, pole angle, angular velocity).
Output: 2 (left, right actions).
"""

import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.bpu import BPU


def run_cartpole(adjacency, name="unnamed", n_episodes=500, lr=1e-3,
                 device="cuda", seed=42):
    """Train a BPU policy network on CartPole-v1 using REINFORCE.

    Args:
        adjacency: (N, N) numpy array or scipy sparse matrix.
        name: Human-readable name for this connectome/configuration.
        n_episodes: Number of training episodes.
        lr: Learning rate for Adam optimizer.
        device: Torch device string.
        seed: Random seed for reproducibility.

    Returns:
        dict with keys: name, task, n_neurons, mean_reward_last_50,
        max_reward, final_loss, train_time_sec, learnable_params,
        fixed_params, n_episodes, lr, seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    try:
        import gymnasium as gym
    except ImportError:
        import gym

    env = gym.make("CartPole-v1")
    env.reset(seed=seed)

    # --- Model ---
    model = BPU(adjacency, d_in=4, d_out=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    gamma = 0.99

    # --- Train with REINFORCE ---
    t0 = time.time()
    all_rewards = []
    final_loss = 0.0

    for episode in range(n_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            logits = model(state_tensor)
            probs = torch.softmax(logits, dim=1)
            dist = Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            state, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

        all_rewards.append(sum(rewards))

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        policy_loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        final_loss = policy_loss.item()

    train_time = time.time() - t0
    env.close()

    # Compute summary statistics
    last_50 = all_rewards[-50:] if len(all_rewards) >= 50 else all_rewards
    mean_reward_last_50 = np.mean(last_50)
    max_reward = max(all_rewards)

    return {
        "name": name,
        "task": "CartPole",
        "n_neurons": model.N,
        "mean_reward_last_50": round(float(mean_reward_last_50), 2),
        "max_reward": float(max_reward),
        "accuracy": round(float(mean_reward_last_50) / 500.0, 4),  # normalized score
        "final_loss": final_loss,
        "train_time_sec": round(train_time, 2),
        "learnable_params": model.count_learnable_params(),
        "fixed_params": model.count_fixed_params(),
        "n_episodes": n_episodes,
        "epochs": n_episodes,  # alias for consistency
        "lr": lr,
        "seed": seed,
    }
