"""Streamlit dashboard for DQN agent training and visualization."""

from __future__ import annotations

import streamlit as st

from src.data import ENV_IDS, make_env, get_env_info, run_episode
from src.model import DQNAgent
from src.visualizer import plot_epsilon_decay, plot_rewards

st.set_page_config(page_title="DQN Agent", layout="wide")
st.title("Deep Q-Network (DQN) Agent")

tab1, tab2, tab3 = st.tabs(["Train", "Evaluate", "Theory"])

with tab1:
    st.header("Train DQN Agent")
    col1, col2, col3 = st.columns(3)
    with col1:
        env_choice = st.selectbox("Environment", list(ENV_IDS.keys()))
        episodes = st.slider("Episodes", 50, 1000, 200, 50)
    with col2:
        lr = st.select_slider("Learning Rate", options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3)
        gamma = st.slider("Discount (gamma)", 0.8, 1.0, 0.99, 0.01)
    with col3:
        hidden = st.slider("Hidden Dim", 32, 256, 128, 32)
        buffer = st.slider("Buffer Size", 1000, 50000, 10000, 1000)

    if st.button("Start Training", type="primary"):
        env = make_env(env_choice)
        info = get_env_info(env)
        agent = DQNAgent(
            state_dim=info["state_dim"],
            action_dim=info["action_dim"],
            hidden_dim=hidden,
            lr=lr,
            gamma=gamma,
            buffer_capacity=buffer,
        )

        progress = st.progress(0)
        reward_placeholder = st.empty()
        chart_placeholder = st.empty()

        for ep in range(1, episodes + 1):
            reward, steps = run_episode(env, agent)
            agent.episode_rewards.append(reward)
            progress.progress(ep / episodes)
            if ep % 10 == 0:
                avg = sum(agent.episode_rewards[-10:]) / 10
                reward_placeholder.metric(f"Episode {ep}", f"Reward: {reward:.1f}", f"Avg10: {avg:.1f}")

        env.close()
        progress.empty()
        st.success(f"Training complete! Best reward: {max(agent.episode_rewards):.1f}")
        plot_rewards(agent.episode_rewards, save_path="/tmp/dqn_rewards.png")
        st.image("/tmp/dqn_rewards.png")

with tab2:
    st.header("Evaluate Agent")
    st.info("Use CLI to evaluate a saved model: `python -m src.main evaluate --model-path model.pt`")

with tab3:
    st.header("How DQN Works")
    st.markdown("""
    **Deep Q-Network (DQN)** combines Q-Learning with deep neural networks:

    1. **Q-Function**: Estimates expected future reward for each action in a given state
    2. **Experience Replay**: Stores transitions (s, a, r, s') and samples randomly to break correlation
    3. **Target Network**: Separate frozen network for stable TD targets, updated periodically
    4. **Epsilon-Greedy**: Balances exploration (random actions) with exploitation (best known action)

    **Key Equation**: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
    """)
    st.subheader("Epsilon Decay")
    plot_epsilon_decay(1.0, 0.01, 0.995, save_path="/tmp/epsilon.png")
    st.image("/tmp/epsilon.png")
