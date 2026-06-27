## Project Overview

This project implements Multi-Agent Deep Deterministic Policy Gradient (MADDPG) for a robot soccer environment. It involves training multiple agents to play soccer cooperatively.

## High-Level Code Architecture and Structure

- `**robot_soccer_env.py**`: Defines the `RobotSoccerEnv` using `gymnasium` and `pettingzoo`, providing a multi-agent environment for soccer.
- `**agents/maddpg_agent.py**`: Contains the core MADDPG algorithm implementation, including actor and critic networks, replay buffer, and learning logic.
- `**configs/config.py**`: Centralized configuration for the environment, training parameters, and MADDPG hyperparameters.
- `**train_maddpg.py**`: The main script to run the MADDPG training process, utilizing the environment, agent, and configuration. It handles the training loop, logging, and model saving.
- `**app.py**`: A Streamlit application that provides a dashboard for monitoring training progress, adjusting some hyperparameters (though full integration with `train_maddpg.py` requires more setup), and visualizing trained agents playing a demo match.
- `**maddpg_logs/**`: Directory for saving training logs.
- `**maddpg_model/**`: Directory for saving trained agent models.
- `**utils/**`: Likely contains utility functions or components used by agents or the environment.
- `**visualization/streamlit_renderer.py**`: Used by `app.py` to render the soccer animation within the Streamlit dashboard.

## Simulation Animation Results

The project provides a Streamlit dashboard (`app.py`) for visualizing the robot soccer environment and watching trained agents play.

To view simulation animations:

1. **Run the Streamlit Dashboard:**
  ```bash
    streamlit run app.py
  ```
2. In the Streamlit application that opens in your browser:
  - Navigate to the "📺 Watch Matches" tab.
  - Ensure "Show Live Animation" is checked in the sidebar.
  - Click the "▶️ Start Demo Match" button to see the agents play in the simulated environment.

If you have trained a model by running `python train_maddpg.py`, you can also check "Use Trained Agents for Demo" in the sidebar to visualize the performance of your trained agents.

## Commonly Used Commands

- **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
- **Run MADDPG Training:**
  ```bash
  python train_maddpg.py
  ```
  This script accepts optional arguments:
  - `--timesteps`: Total training timesteps (e.g., `python train_maddpg.py --timesteps 500000`)
  - `--log_interval`: Frequency (in steps) to log training progress.
  - `--save_interval`: Frequency (in steps) to save model checkpoints.
- **Run Streamlit Dashboard (for visualization and control):**
  ```bash
  streamlit run app.py
  ```
  ### Demo Match
  ![Demo Match](assets\image.jpg)

