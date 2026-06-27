# app.py
"""Streamlit dashboard for Robot Soccer League - MADDPG Training Control & Visualization."""

import streamlit as st
import numpy as np
import pandas as pd
from streamlit.components.v1 import html
import time
import os

# Import the environment (for visualization)
from robot_soccer_env import RobotSoccerEnv
from agents.maddpg_agent import MADDPG
from configs.config import config
from visualization.streamlit_renderer import render_soccer_animation
import io

# Helper function to load MADDPG agent
@st.cache_resource
def load_maddpg_agent():
    env = RobotSoccerEnv(max_cycles=config.MAX_CYCLES)
    observations, _ = env.reset()  # Initialize the environment first!
    agent_ids = env.agents
    num_agents = len(agent_ids)

    obs_dim = env.observation_spaces[agent_ids[0]].shape[0]
    act_dim_cont = 2  # linear_vel, angular_vel
    act_dim_disc = 2  # kick (0 or 1)

    maddpg = MADDPG(
        obs_dim_per_agent=obs_dim,
        action_dim_continuous_per_agent=act_dim_cont,
        action_dim_discrete_per_agent=act_dim_disc,
        num_agents=num_agents,
        env_action_spaces_dict=env.action_spaces,
        device=config.DEVICE,
        seed=config.SEED
    )

    model_path = os.path.join(config.MODEL_SAVE_PATH, "final_model")
    if os.path.exists(model_path + "_agent_0_actor.pt"):
        try:
            maddpg.load(folder=config.MODEL_SAVE_PATH, prefix="final_model")
            st.sidebar.success("Trained MADDPG model loaded!")
            return maddpg
        except Exception as e:
            st.sidebar.error(f"Error loading trained model: {e}")
            return None
    else:
        st.sidebar.warning("No trained MADDPG model found. Run training first!")
        return None

# Page config
st.set_page_config(
    page_title="🤖 Robot Soccer League - MADDPG Training",
    page_icon="⚽",
    layout="wide"
)

# Title
st.title("🤖 Robot Soccer League - MADDPG Training Dashboard")
st.markdown("---")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Training Controls")

    total_timesteps = st.number_input(
        "Total Timesteps",
        min_value=10000,
        max_value=10000000,
        value=2000000,
        step=100000
    )

    batch_size = st.slider("Batch Size", 32, 512, 256, 32)
    lr_actor = st.select_slider("Actor LR", options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3], value=1e-4)
    lr_critic = st.select_slider("Critic LR", options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3], value=1e-3)

    st.markdown("---")
    st.header("🎮 Live Visualization")

    show_animation = st.checkbox("Show Live Animation", value=True)
    use_trained_agents = st.checkbox("Use Trained Agents for Demo", value=False)
    render_fps = st.slider("Render FPS", 5, 60, 30, 5)

# Load the MADDPG agent (cached)
maddpg_agent = None
if use_trained_agents:
    maddpg_agent = load_maddpg_agent()

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏃 Training", "📊 Metrics", "📺 Watch Matches", "ℹ️ Info"])

# Tab 1: Training Control
with tab1:
    st.subheader("Start Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Start Training", type="primary"):
            st.info("Training would start here (integration with train_maddpg.py)")
            st.warning("Note: Full Streamlit training integration requires additional setup")
    
    with col2:
        if st.button("⏹️ Stop Training"):
            st.info("Training stopped")
    
    # Training progress simulation
    st.markdown("### 📈 Training Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate training
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"Training step: {i * 20000}/{total_timesteps}")
        time.sleep(0.01)
    
    status_text.text("Training complete!")

# Tab 2: Metrics
with tab2:
    st.subheader("Training Metrics")
    
    # Generate sample data
    episodes = list(range(1, 101))
    rewards = np.random.randn(100).cumsum()
    
    # Plot rewards
    chart_data = pd.DataFrame({
        'Episode': episodes,
        'Average Reward': rewards
    })
    
    st.line_chart(chart_data.set_index('Episode'))
    
    # Metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Episodes Completed", "100", "+10")
    with col2:
        st.metric("Avg Reward", f"{rewards[-1]:.2f}", f"{rewards[-1] - rewards[-10]:.2f}")
    with col3:
        st.metric("Win Rate", "78%", "+5%")
    with col4:
        st.metric("Goals Scored", "42", "+3")

# Tab 3: Watch Matches
with tab3:
    st.subheader("Watch Trained Agents Play")

    # Animation settings
    num_steps = st.slider("Animation Duration (steps)", 20, 200, 100)
    animation_placeholder = st.empty()

    if show_animation:
        # Simulate match and show animation
        if st.button("▶️ Start Demo Match"):
            if use_trained_agents and maddpg_agent is None:
                st.warning("Cannot start demo: No trained agent model loaded.")
            else:
                with st.spinner("Generating animation..."):
                    # Create environment and generate frames
                    env = RobotSoccerEnv(max_cycles=num_steps + 50)
                    frames = render_soccer_animation(env, maddpg_agent=maddpg_agent, num_steps=num_steps, fps=render_fps)
                    env.close()

                    # Display frames as animation
                    animation_display = st.empty()
                    for frame_idx, frame in enumerate(frames):
                        animation_display.image(frame, caption=f"Step {frame_idx + 1}/{len(frames)}")
                        time.sleep(1 / render_fps)

                    st.success("Demo match completed!")
    else:
        st.info("Enable 'Show Live Animation' in sidebar to watch matches")

# Tab 4: Info
with tab4:
    st.subheader("Project Information")
    
    st.markdown("""
    ### 🤖 Robot Soccer League - MADDPG
    
    **Algorithm:** Multi-Agent Deep Deterministic Policy Gradient
    
    **Agents:** 10 total (5 vs 5)
    - Team 0 (Blue): Agents 0-4
    - Team 1 (Red): Agents 5-9
    
    **Actions:**
    - `linear_vel`: Forward/backward movement [-1, 1]
    - `angular_vel`: Turning speed [-1, 1]
    - `kick`: Discrete kick action [0, 1]
    
    **Observations:**
    - Self position, orientation, velocity
    - Ball position, velocity
    - All teammates and opponents positions
    
    **Rewards:**
    - Scoring goal: +10.0
    - Conceding goal: -10.0
    - Successful pass: +1.0
    - Ball control: +0.05 per step
    - Interception: +2.0
    
    **Files:**
    - `train_maddpg.py`: Main training script
    - `visualize_training.py`: Pygame visualization
    - `robot_soccer_env.py`: Environment definition
    """)
    
    # Download trained model
    st.markdown("---")
    st.subheader("📥 Download Model")
    st.info("Trained model files will be available here after training completes")

# Footer
st.markdown("---")
st.markdown("<center>🤖 Robot Soccer League | Powered by MADDPG | Built with Streamlit</center>", unsafe_allow_html=True)