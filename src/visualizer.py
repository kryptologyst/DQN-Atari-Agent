"""Visualization utilities for DQN training progress."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_rewards(
    rewards: list[float],
    window: int = 10,
    save_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.3, color="steelblue", label="Episode Reward")
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            range(window - 1, len(rewards)),
            smoothed,
            color="darkorange",
            linewidth=2,
            label=f"{window}-Episode Moving Avg",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("DQN Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Reward plot saved to %s", save_path)
    plt.close(fig)


def plot_epsilon_decay(
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    steps: int = 1000,
    save_path: str | None = None,
) -> None:
    eps = epsilon_start
    values = [eps]
    for _ in range(steps - 1):
        eps = max(epsilon_end, eps * epsilon_decay)
        values.append(eps)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(values, color="purple")
    ax.set_xlabel("Step")
    ax.set_ylabel("Epsilon")
    ax.set_title("Epsilon Decay Schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Epsilon decay plot saved to %s", save_path)
    plt.close(fig)


def plot_q_values(
    q_values: list[float],
    save_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(q_values, color="teal", linewidth=1)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Average Q-Value")
    ax.set_title("Q-Value Convergence")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Q-value plot saved to %s", save_path)
    plt.close(fig)
