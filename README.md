# DQN Atari Agent

**Deep Q-Network** reinforcement learning agent with experience replay and target networks.

## Overview

- DQN with configurable hidden layers
- Experience replay buffer for stable training
- Target network with periodic updates
- Epsilon-greedy exploration with decay
- Supports CartPole, MountainCar, Acrobot, LunarLander
- **Streamlit dashboard** with live training visualization

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
# CLI: python -m src.main train --env-name cartpole --episodes 300
# CLI: python -m src.main evaluate --model-path model.pt
pytest tests/ -v
```

## Docker

```bash
docker compose up --build
```

## License

MIT
# DQN-Atari-Agent
