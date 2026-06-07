"""Tests for DQN agent."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data import ENV_IDS, make_env, get_env_info
from src.model import DQN, DQNAgent, ReplayBuffer


class TestDQN:
    def test_forward_shape(self) -> None:
        net = DQN(state_dim=4, action_dim=2, hidden_dim=64)
        x = torch.randn(1, 4)
        out = net(x)
        assert out.shape == (1, 2)

    def test_forward_batch(self) -> None:
        net = DQN(state_dim=4, action_dim=2, hidden_dim=64)
        x = torch.randn(32, 4)
        out = net(x)
        assert out.shape == (32, 2)


class TestReplayBuffer:
    def test_push_and_sample(self) -> None:
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(np.array([i]), 0, 1.0, np.array([i + 1]), False)
        assert len(buf) == 50
        batch = buf.sample(16)
        assert len(batch) == 16

    def test_sample_limited_by_buffer(self) -> None:
        buf = ReplayBuffer(capacity=100)
        buf.push(np.array([0]), 0, 1.0, np.array([1]), False)
        batch = buf.sample(32)
        assert len(batch) == 1


class TestDQNAgent:
    def test_init(self) -> None:
        agent = DQNAgent(state_dim=4, action_dim=2)
        assert agent.epsilon == 1.0
        assert agent.steps_done == 0

    def test_select_action_returns_int(self) -> None:
        agent = DQNAgent(state_dim=4, action_dim=2)
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = agent.select_action(state, evaluate=True)
        assert isinstance(action, int)
        assert 0 <= action < 2

    def test_update_noop_when_buffer_small(self) -> None:
        agent = DQNAgent(state_dim=4, action_dim=2, batch_size=64)
        loss = agent.update()
        assert loss is None

    def test_save_and_load(self, tmp_path) -> None:
        agent = DQNAgent(state_dim=4, action_dim=2)
        agent.episode_rewards = [10.0, 20.0]
        path = str(tmp_path / "agent.pt")
        agent.save(path)
        agent2 = DQNAgent(state_dim=4, action_dim=2)
        agent2.load(path)
        assert agent2.episode_rewards == [10.0, 20.0]


class TestEnvUtils:
    def test_make_env_cartpole(self) -> None:
        env = make_env("cartpole")
        info = get_env_info(env)
        assert info["state_dim"] == 4
        assert info["action_dim"] == 2
        env.close()

    def test_env_ids_has_expected_keys(self) -> None:
        for key in ["cartpole", "mountaincar", "acrobot", "lunarlander"]:
            assert key in ENV_IDS
