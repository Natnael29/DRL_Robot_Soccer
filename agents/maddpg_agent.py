# agents/maddpg_agent.py
"""Implementation of a Multi-Agent Deep Deterministic Policy Gradient (MADDPG) agent.

This implementation assumes:
*   A centralized critic that receives concatenated observations and continuous actions for all agents.
*   Independent actor networks per agent that output deterministic continuous actions (scaled to [-1, 1]).
*   Discrete actions (e.g., the 'kick' action) are sampled randomly during exploration; they are not learned directly.

The code is intentionally written to be clear and self‑contained for a learning project. It uses the
*MLP* class from ``agents.mlp_networks`` for both actors and the critic.

Key components:
*   **Actor networks** – one per agent, mapping observations to continuous actions.
*   **Centralized critic** – a single network that evaluates the joint state‑action pair and outputs a Q‑value for each agent.
*   **Replay buffer** – stores full multi‑agent transitions and samples mini‑batches for training.
*   **Training loop** – computes target Q‑values, updates the critic with a MSE loss, and updates each actor to maximize its Q‑value.

The implementation follows the standard MADDPG algorithm described in the original paper (Lowe et al., 2017).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import os
from collections import deque

# Local imports – ensure the package structure exists.
from configs.config import config
from utils.replay_buffer import MultiAgentReplayBuffer
from agents.mlp_networks import MLP

class MADDPG:
    """MADDPG agent suite for the Robot Soccer environment.

    This class manages a set of actor networks (one per agent) and a single centralized critic.
    It provides methods for action selection, experience storage, learning updates, and model persistence.
    """

    def __init__(self, obs_dim_per_agent, action_dim_continuous_per_agent,
                 action_dim_discrete_per_agent, num_agents, env_action_spaces_dict,
                 device='cpu', seed=0):
        """Initialize MADDPG networks, optimizers, replay buffer, and exploration noise.

        Args:
            obs_dim_per_agent (int): Observation vector size for a single agent.
            action_dim_continuous_per_agent (int): Number of continuous action dimensions (e.g., 2 for linear/angular velocity).
            action_dim_discrete_per_agent (int): Number of discrete actions (e.g., 1 for the 'kick' action).
            num_agents (int): Total agents (team_size * 2).
            env_action_spaces_dict (dict): Mapping ``agent_id -> action_space`` (PettingZoo Dict spaces).
            device (str): ``'cpu'`` or ``'cuda'``.
            seed (int): Random seed for reproducibility.
        """
        self.obs_dim = obs_dim_per_agent
        self.act_dim_cont = action_dim_continuous_per_agent
        self.act_dim_disc = action_dim_discrete_per_agent
        self.num_agents = num_agents
        self.env_action_spaces = env_action_spaces_dict
        self.device = torch.device(device)
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Hyper‑parameters from the global config.
        self.gamma = config.GAMMA
        self.tau = config.TAU
        self.batch_size = config.BATCH_SIZE
        self.buffer_size = config.BUFFER_SIZE
        self.lr_actor = config.LR_ACTOR
        self.lr_critic = config.LR_CRITIC
        self.update_every = config.UPDATE_EVERY
        self.max_expl_noise = config.MAX_EXPLORATION_NOISE
        self.min_expl_noise = config.MIN_EXPLORATION_NOISE
        self.expl_noise_decay = config.EXPLORATION_NOISE_DECAY
        self.current_expl_noise = self.max_expl_noise

        # ---------------------------------------------------------------------
        #   Networks
        # ---------------------------------------------------------------------
        # Actors – one per agent (deterministic policy for continuous actions).
        self.actors = nn.ModuleDict()
        self.actors_target = nn.ModuleDict()
        self.actor_optimizers = {}
        for agent_id in self.env_action_spaces.keys():
            actor = MLP(input_dim=self.obs_dim,
                        output_dim=self.act_dim_cont,
                        hidden_dims=config.HIDDEN_DIMS,
                        activation=nn.ReLU,
                        final_activation=nn.Tanh)  # Output scaled to [-1, 1]
            actor_target = copy.deepcopy(actor)
            self.actors[agent_id] = actor.to(self.device)
            self.actors_target[agent_id] = actor_target.to(self.device)
            self.actor_optimizers[agent_id] = optim.Adam(self.actors[agent_id].parameters(), lr=self.lr_actor)

        # Centralized Critic – takes concatenated observations and continuous actions of ALL agents.
        # Input dimension = (num_agents * obs_dim) + (num_agents * act_dim_cont)
        critic_input_dim = (self.num_agents * self.obs_dim) + (self.num_agents * self.act_dim_cont)
        self.critic = MLP(input_dim=critic_input_dim,
                           output_dim=self.num_agents,  # One Q‑value per agent
                           hidden_dims=config.HIDDEN_DIMS,
                           activation=nn.ReLU).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # ---------------------------------------------------------------------
        #   Replay Buffer (shared among all agents)
        # ---------------------------------------------------------------------
        self.replay_buffer = MultiAgentReplayBuffer(buffer_size=self.buffer_size,
                                                    batch_size=self.batch_size,
                                                    num_agents=self.num_agents,
                                                    seed=self.seed,
                                                    device=self.device)
        self.learn_step_counter = 0

    # ---------------------------------------------------------------------
    #   Utility methods
    # ---------------------------------------------------------------------
    def _soft_update(self, target, source, tau):
        """Soft‑update target network parameters.
        target ← τ * source + (1‑τ) * target
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def _hard_update(self, target, source):
        """Copy weights from source to target (hard update)."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    # ---------------------------------------------------------------------
    #   Action selection (with optional exploration noise)
    # ---------------------------------------------------------------------
    def select_actions(self, obs_dict, explore=True):
        """Select actions for all agents given their observations.

        Args:
            obs_dict (dict): ``agent_id -> observation`` (numpy arrays).
            explore (bool): Whether to add exploration noise to continuous actions.
        Returns:
            dict: ``agent_id -> action`` where the action matches the environment's ``Dict`` space.
        """
        actions = {}
        for agent_id, obs in obs_dict.items():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)  # Shape (1, obs_dim)
            # Continuous actions from actor
            with torch.no_grad():
                cont_action = self.actors[agent_id](obs_tensor).squeeze(0)  # Shape (act_dim_cont,)
            if explore:
                noise = torch.randn_like(cont_action) * self.current_expl_noise
                cont_action = cont_action + noise
                cont_action = torch.clamp(cont_action, config.ACTION_BOUND[0], config.ACTION_BOUND[1])
            cont_action_np = cont_action.cpu().numpy()

            # Discrete action (kick) – sampled randomly during exploration.
            # In a full implementation this could be learned via a separate policy.
            if 'kick' in self.env_action_spaces[agent_id].spaces:
                kick_space = self.env_action_spaces[agent_id].spaces['kick']
                discrete_action = kick_space.sample() if explore else 0
            else:
                discrete_action = 0

            actions[agent_id] = {
                'linear_vel': np.array([cont_action_np[0]], dtype=np.float32),
                'angular_vel': np.array([cont_action_np[1]], dtype=np.float32),
                'kick': discrete_action
            }
        return actions

    # ---------------------------------------------------------------------
    #   Experience storage
    # ---------------------------------------------------------------------
    def store_transition(self, states, actions, rewards, next_states, done):
        """Add a full multi‑agent transition to the replay buffer.
        ``states`` / ``next_states`` are dictionaries ``agent_id -> np.ndarray``.
        ``actions`` is a dictionary ``agent_id -> dict`` (matching the env action space).
        ``rewards`` is a dictionary ``agent_id -> float``.
        ``done`` is a boolean indicating episode termination.
        """
        # Convert to ordered lists based on the internal actor order.
        agent_ids = list(self.actors.keys())
        states_list = [states[aid] for aid in agent_ids]
        next_states_list = [next_states[aid] for aid in agent_ids]
        # For continuous actions, extract the two velocity components.
        actions_cont_list = []
        for aid in agent_ids:
            act_dict = actions[aid]
            # Each component is a 1‑D array; flatten to a scalar.
            lin = act_dict['linear_vel'][0]
            ang = act_dict['angular_vel'][0]
            actions_cont_list.append(np.array([lin, ang], dtype=np.float32))
        rewards_list = [rewards[aid] for aid in agent_ids]
        self.replay_buffer.add(states_list, actions_cont_list, rewards_list, next_states_list, done)

    # ---------------------------------------------------------------------
    #   Learning step (updates both actor(s) and critic)
    # ---------------------------------------------------------------------
    def learn(self):
        """Sample a batch from the replay buffer and perform a gradient update.
        The method follows the MADDPG update rules:
        *   Compute target Q‑values using the target critic and target actors.
        *   Update the centralized critic with MSE loss.
        *   Update each actor to maximise its Q‑value (policy gradient).
        *   Soft‑update all target networks.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough data yet.

        # Sample a batch – each element is a list (per agent) of tensors.
        (states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch) = self.replay_buffer.sample()
        # states_batch / next_states_batch: list of length num_agents (each element shape (batch, obs_dim))
        # actions_batch: list of length num_agents (each element shape (batch, act_dim_cont))
        # rewards_batch: list of length num_agents (each element shape (batch, 1))
        # dones_batch: (batch, 1)

        # -----------------------------------------------------------------
        #   1️⃣ Compute target actions for next states using target actors.
        # -----------------------------------------------------------------
        target_actions_next = []
        for i, agent_id in enumerate(self.actors_target.keys()):
            with torch.no_grad():
                target_act = self.actors_target[agent_id](next_states_batch[i])  # Shape (batch, act_dim_cont)
                target_actions_next.append(target_act)
        # Concatenate next states and target actions for the target critic.
        # Combined input shape: (batch, num_agents * obs_dim + num_agents * act_dim_cont)
        # Helper: concatenate along last dimension.
        next_states_concat = torch.cat(next_states_batch, dim=1)  # (batch, num_agents * obs_dim)
        target_acts_concat = torch.cat(target_actions_next, dim=1)  # (batch, num_agents * act_dim_cont)
        critic_target_input = torch.cat((next_states_concat, target_acts_concat), dim=1)

        with torch.no_grad():
            q_next = self.critic_target(critic_target_input)  # Shape (batch, num_agents)
            # Compute target Q for each agent: r_i + γ * q_next_i * (1 - done)
            # Ensure rewards and dones are column vectors to avoid broadcasting issues.
            dones = dones_batch.unsqueeze(1)  # Shape (batch, 1)
            target_q = []
            for i in range(self.num_agents):
                # rewards_batch[i] has shape (batch,). Convert to column vector.
                ri = rewards_batch[i].unsqueeze(1)  # Shape (batch, 1)
                qi_next = q_next[:, i].unsqueeze(1)  # Shape (batch, 1)
                target_q_i = ri + self.gamma * qi_next * (1.0 - dones)
                target_q.append(target_q_i)
            target_q_stacked = torch.cat(target_q, dim=1)  # (batch, num_agents)

        # -----------------------------------------------------------------
        #   2️⃣ Compute current Q‑values with the main critic.
        # -----------------------------------------------------------------
        states_concat = torch.cat(states_batch, dim=1)
        actions_concat = torch.cat(actions_batch, dim=1)
        critic_input = torch.cat((states_concat, actions_concat), dim=1)
        current_q = self.critic(critic_input)  # (batch, num_agents)

        # Critic loss (MSE)
        loss_critic = F.mse_loss(current_q, target_q_stacked)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # -----------------------------------------------------------------
        #   3️⃣ Update each actor to maximise its own Q‑value.
        # -----------------------------------------------------------------
        # For each agent, compute actions from its current actor (others use current actions as well).
        actor_actions = []
        for i, agent_id in enumerate(self.actors.keys()):
            # Use the current observations for each agent.
            act = self.actors[agent_id](states_batch[i])
            actor_actions.append(act)
        # Construct critic input with current states and *actor* actions.
        actor_actions_concat = torch.cat(actor_actions, dim=1)
        critic_input_for_actor = torch.cat((states_concat, actor_actions_concat), dim=1)
        # Evaluate Q‑values for the joint action.
        q_vals = self.critic(critic_input_for_actor)  # (batch, num_agents)
        # Actor loss = - mean Q_i for its own index.
        actor_losses = []
        for idx, agent_id in enumerate(self.actors.keys()):
            # Negative because we want to maximize Q.
            loss_actor_i = -q_vals[:, idx].mean()
            actor_losses.append(loss_actor_i)
        total_actor_loss = sum(actor_losses)
        # Zero all actor optimizers
        for opt in self.actor_optimizers.values():
            opt.zero_grad()
        # Single backward pass through the shared graph
        total_actor_loss.backward()
        # Clip gradients and step each actor optimizer
        for agent_id in self.actors.keys():
            torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), max_norm=1.0)
            self.actor_optimizers[agent_id].step()
        loss_actor_total = sum(loss.item() for loss in actor_losses)
        # (loss_actor_total is optional for logging)

        # -----------------------------------------------------------------
        #   4️⃣ Soft‑update target networks.
        # -----------------------------------------------------------------
        for agent_id in self.actors.keys():
            self._soft_update(self.actors_target[agent_id], self.actors[agent_id], self.tau)
        self._soft_update(self.critic_target, self.critic, self.tau)

        # -----------------------------------------------------------------
        #   5️⃣ Decay exploration noise.
        # -----------------------------------------------------------------
        self.current_expl_noise = max(self.min_expl_noise,
                                      self.current_expl_noise * self.expl_noise_decay)

    # ---------------------------------------------------------------------
    #   Model persistence – save / load
    # ---------------------------------------------------------------------
    def save(self, folder="maddpg_model", prefix="maddpg"):
        """Save the actor and critic parameters to ``folder``.
        The files will be named ``<prefix>_critic.pt`` and ``<prefix>_<agent>_actor.pt``.
        """
        os.makedirs(folder, exist_ok=True)
        torch.save(self.critic.state_dict(), os.path.join(folder, f"{prefix}_critic.pt"))
        torch.save(self.critic_target.state_dict(), os.path.join(folder, f"{prefix}_critic_target.pt"))
        for agent_id, actor in self.actors.items():
            torch.save(actor.state_dict(), os.path.join(folder, f"{prefix}_{agent_id}_actor.pt"))
            torch.save(self.actors_target[agent_id].state_dict(), os.path.join(folder, f"{prefix}_{agent_id}_actor_target.pt"))
        print(f"MADDPG parameters saved to {folder}")

    def load(self, folder="maddpg_model", prefix="maddpg"):
        """Load parameters from ``folder``.
        Expected files are the same as saved by :meth:`save`.
        """
        self.critic.load_state_dict(torch.load(os.path.join(folder, f"{prefix}_critic.pt"), map_location=self.device))
        self.critic_target.load_state_dict(torch.load(os.path.join(folder, f"{prefix}_critic_target.pt"), map_location=self.device))
        for agent_id in self.actors.keys():
            self.actors[agent_id].load_state_dict(torch.load(os.path.join(folder, f"{prefix}_{agent_id}_actor.pt"), map_location=self.device))
            self.actors_target[agent_id].load_state_dict(torch.load(os.path.join(folder, f"{prefix}_{agent_id}_actor_target.pt"), map_location=self.device))
        print(f"MADDPG parameters loaded from {folder}")

    # ---------------------------------------------------------------------
    #   Helper: reset exploration noise (useful after evaluation phases)
    # ---------------------------------------------------------------------
    def reset_exploration(self):
        """Reset exploration noise to its initial maximum value."""
        self.current_expl_noise = self.max_expl_noise
