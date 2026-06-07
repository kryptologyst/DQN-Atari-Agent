"""CLI for DQN agent training and evaluation."""

from __future__ import annotations

import logging

import typer

from src.data import ENV_IDS, make_env, get_env_info, run_episode
from src.model import DQNAgent
from src.visualizer import plot_rewards

app = typer.Typer(help="DQN Atari Agent CLI")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@app.command()
def train(
    env_name: str = typer.Option("cartpole", help="Environment: cartpole, mountaincar, acrobot, lunarlander"),
    episodes: int = typer.Option(300, help="Number of training episodes"),
    lr: float = typer.Option(1e-3, help="Learning rate"),
    gamma: float = typer.Option(0.99, help="Discount factor"),
    save_path: str | None = typer.Option(None, help="Save trained model"),
    plot_path: str | None = typer.Option(None, help="Save reward plot"),
) -> None:
    env = make_env(env_name)
    info = get_env_info(env)
    logger.info("State dim: %d, Action dim: %d", info["state_dim"], info["action_dim"])

    agent = DQNAgent(
        state_dim=info["state_dim"],
        action_dim=info["action_dim"],
        lr=lr,
        gamma=gamma,
    )

    for ep in range(1, episodes + 1):
        reward, steps = run_episode(env, agent)
        agent.episode_rewards.append(reward)
        if ep % 20 == 0:
            avg = sum(agent.episode_rewards[-20:]) / 20
            logger.info("Ep %d/%d | Reward: %.1f | Avg20: %.1f | Epsilon: %.3f", ep, episodes, reward, avg, agent.epsilon)

    env.close()
    logger.info("Training complete. Best reward: %.1f", max(agent.episode_rewards))

    if save_path:
        agent.save(save_path)
    if plot_path:
        plot_rewards(agent.episode_rewards, save_path=plot_path)


@app.command()
def evaluate(
    model_path: str = typer.Option(..., help="Path to saved model"),
    env_name: str = typer.Option("cartpole", help="Environment"),
    episodes: int = typer.Option(10, help="Evaluation episodes"),
    render: bool = typer.Option(False, help="Render environment"),
) -> None:
    env = make_env(env_name, render_mode="human" if render else None)
    info = get_env_info(env)
    agent = DQNAgent(state_dim=info["state_dim"], action_dim=info["action_dim"])
    agent.load(model_path)

    rewards = []
    for ep in range(1, episodes + 1):
        reward, steps = run_episode(env, agent, evaluate=True)
        rewards.append(reward)
        logger.info("Ep %d: Reward %.1f, Steps %d", ep, reward, steps)

    env.close()
    logger.info("Avg reward over %d episodes: %.1f", episodes, sum(rewards) / len(rewards))


if __name__ == "__main__":
    app()
