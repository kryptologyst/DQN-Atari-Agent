"""Environment utilities for DQN training."""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)

ENV_IDS = {
    "cartpole": "CartPole-v1",
    "mountaincar": "MountainCar-v0",
    "acrobot": "Acrobot-v1",
    "lunarlander": "LunarLander-v2",
}


def make_env(env_id: str = "cartpole", render_mode: str | None = None) -> gym.Env:
    full_id = ENV_IDS.get(env_id, ENV_IDS["cartpole"])
    env = gym.make(full_id, render_mode=render_mode)
    logger.info("Created environment: %s", full_id)
    return env


def get_env_info(env: gym.Env) -> dict[str, Any]:
    return {
        "state_dim": env.observation_space.shape[0],
        "action_dim": env.action_space.n,
        "action_space": str(env.action_space),
        "obs_space": str(env.observation_space),
    }


def run_episode(
    env: gym.Env,
    agent: Any,
    max_steps: int = 500,
    render: bool = False,
    evaluate: bool = False,
) -> tuple[float, int]:
    state, _ = env.reset()
    total_reward = 0.0
    for step in range(max_steps):
        action = agent.select_action(state, evaluate=evaluate)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if not evaluate:
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
        state = next_state
        total_reward += float(reward)
        if done:
            break
    return total_reward, step + 1
