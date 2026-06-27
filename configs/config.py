# configs/config.py
import torch
import numpy as np
import random
import os

class Config:
    def __init__(self):
        # --- Environment Settings ---
        self.ENV_ID = "RobotSoccerEnv-v0" # For reference, custom envs are often instantiated directly
        self.MAX_CYCLES = 500              # Max steps per episode in env
        self.TEAM_SIZE = 5                 # Number of agents per team (total agents = TEAM_SIZE * 2)

        # --- Training Settings ---
        self.TOTAL_TIMESTEPS = int(2e6)    # Total training timesteps (adjust as needed)
        self.SEED = 42                     # Random seed for reproducibility
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # Device for PyTorch
        self.LOG_DIR = os.path.join("maddpg_logs")    # Directory to save logs
        self.MODEL_SAVE_PATH = os.path.join("maddpg_model") # Directory to save model checkpoints

        # --- MADDPG Hyperparameters ---
        self.BUFFER_SIZE = int(1e6)         # Replay buffer size
        self.BATCH_SIZE = 256               # Minibatch size for training
        self.GAMMA = 0.99                   # Discount factor
        self.TAU = 1e-3                     # Soft update coefficient for target networks
        self.LR_ACTOR = 1e-4                # Learning rate for actor optimizer
        self.LR_CRITIC = 1e-3               # Learning rate for critic optimizer
        self.UPDATE_EVERY = 4               # How often to update the network (in training steps)

        # Exploration noise parameters (for continuous actions)
        self.MAX_EXPLORATION_NOISE = 1.0    # Max std deviation for exploration noise
        self.MIN_EXPLORATION_NOISE = 0.05   # Min std deviation for exploration noise
        self.EXPLORATION_NOISE_DECAY = 0.9999 # Decay rate for exploration noise per step

        # Network architecture
        self.HIDDEN_DIMS = (256, 256)       # Dimensions of hidden layers in MLPs

        # --- Agent Specific Dimensions ---
        # These are derived from the environment, but good to have defaults/placeholders
        # We will fetch these dynamically once the environment is instantiated.
        self.OBS_DIM_PER_AGENT = None       # To be set by environment
        self.ACTION_DIM_CONTINUOUS_PER_AGENT = 2 # linear_vel, angular_vel
        self.ACTION_DIM_DISCRETE_PER_AGENT = 2   # kick (0 or 1)
        self.ACTION_BOUND = (-1.0, 1.0)     # Standard for continuous actions like velocity

    def __str__(self):
        return str(self.__dict__)

# Instantiate the config object
config = Config()
