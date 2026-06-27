# train_maddpg.py
"""Training script for MADDPG on Robot Soccer environment.

This script orchestrates the full training loop:
- Initializes environment and agent
- Runs episodes with action selection and learning
- Saves model checkpoints periodically
- Logs training progress
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

# Import environment and agent
from robot_soccer_env import RobotSoccerEnv
from agents.maddpg_agent import MADDPG
from configs.config import config

def set_random_seeds(seed):
    """Set random seeds for reproducibility."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    # For complete reproducibility
    import random
    random.seed(seed)

def main(total_timesteps=2000000, log_interval=1000, save_interval=50000):
    """Main training function."""
    
    # Set seeds
    set_random_seeds(config.SEED)
    
    # Create directories for saving
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    
    # Initialize environment
    print("Initializing Robot Soccer Environment...")
    env = RobotSoccerEnv(max_cycles=config.MAX_CYCLES)
    agent_ids = env.agents
    num_agents = len(agent_ids)
    
    print(f"Environment created with {num_agents} agents")
    print(f"Observation space per agent: {env.observation_spaces[agent_ids[0]].shape[0]}")
    
    # Initialize MADDPG agent
    print("Initializing MADDPG agent...")
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
    
    print(f"Using device: {config.DEVICE}")
    print(f"Starting training for {total_timesteps} timesteps...")
    print("-" * 50)
    
    # Training loop
    timestep = 0
    episode = 0
    
    with tqdm(total=total_timesteps, desc="Training") as pbar:
        while timestep < total_timesteps:
            # Reset environment
            observations, _ = env.reset()
            episode_reward = {aid: 0.0 for aid in agent_ids}
            
            while True:
                # Select actions
                actions = maddpg.select_actions(observations, explore=True)
                
                # Step environment
                next_obs, rewards, terminations, truncations, _ = env.step(actions)
                done = any(terminations.values()) or any(truncations.values())
                
                # Store transition
                maddpg.store_transition(observations, actions, rewards, next_obs, done)
                
                # Accumulate rewards
                for aid, r in rewards.items():
                    episode_reward[aid] += r
                
                # Learn
                if timestep % config.UPDATE_EVERY == 0:
                    maddpg.learn()
                
                observations = next_obs
                timestep += 1
                pbar.update(1)
                
                # Logging
                if timestep % log_interval == 0:
                    avg_reward = np.mean(list(episode_reward.values()))
                    pbar.set_postfix({
                        'AvgReward': f'{avg_reward:.2f}',
                        'Noise': f'{maddpg.current_expl_noise:.3f}'
                    })
                
                # Save model
                if timestep % save_interval == 0:
                    maddpg.save(folder=config.MODEL_SAVE_PATH, prefix=f"checkpoint_step_{timestep}")
                
                if done or timestep >= total_timesteps:
                    episode += 1
                    break
    
    # Final save
    maddpg.save(folder=config.MODEL_SAVE_PATH, prefix="final_model")
    env.close()
    
    print("-" * 50)
    print(f"Training completed after {timestep} timesteps and {episode} episodes")
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MADDPG on Robot Soccer")
    parser.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS, help="Total training timesteps")
    parser.add_argument("--log_interval", type=int, default=1000, help="Log frequency (steps)")
    parser.add_argument("--save_interval", type=int, default=50000, help="Save frequency (steps)")
    
    args = parser.parse_args()
    
    main(
        total_timesteps=args.timesteps,
        log_interval=args.log_interval,
        save_interval=args.save_interval
    )