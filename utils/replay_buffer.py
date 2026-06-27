# utils/replay_buffer.py
import torch
import numpy as np
import random
from collections import deque, namedtuple

# --- Experience definition for multi-agent transitions ---
# Each element in the tuple will be a list of values, one per agent.
MultiAgentExperience = namedtuple("MultiAgentExperience",
                                  field_names=["states", "actions", "rewards", "next_states", "dones"])

class MultiAgentReplayBuffer:
    """Replay buffer for multi-agent RL environments.

    This buffer stores transitions where each field (states, actions, etc.) contains
    data for ALL agents in the environment at once.
    """

    def __init__(self, buffer_size, batch_size, num_agents, seed, device='cpu'):
        """
        Args:
            buffer_size (int): Maximum number of transitions to store.
            batch_size (int): Number of transitions to sample per batch.
            num_agents (int): Total number of agents in the environment.
            seed (int): Random seed.
            device (str): PyTorch device ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.num_agents = num_agents
        random.seed(seed)
        self.seed = seed

    def add(self, states, actions, rewards, next_states, done):
        """
        Add a new multi-agent transition to memory.

        Args:
            states (list): List of observation arrays (one per agent), shape (num_agents,),
                           each element shape (obs_dim,).
            actions (list): List of action arrays (one per agent), shape (num_agents,),
                            each element shape (act_dim_continuous,).
            rewards (list): List of rewards (one per agent), shape (num_agents,).
            next_states (list): List of next observation arrays (one per agent), shape (num_agents,),
                                 each element shape (obs_dim,).
            done (bool): Whether the episode has terminated.

        Returns:
            None
        """
        # Convert to tensors and move to device
        e = MultiAgentExperience(
            states=[torch.from_numpy(s).float().to(self.device) for s in states],
            actions=[torch.from_numpy(a).float().to(self.device) for a in actions],
            rewards=[torch.tensor([r], dtype=torch.float32).to(self.device) for r in rewards],
            next_states=[torch.from_numpy(ns).float().to(self.device) for ns in next_states],
            dones=torch.tensor([float(done)], dtype=torch.float32).to(self.device)
        )
        self.memory.append(e)

    def sample(self):
        """
        Randomly sample a batch of experiences.

        Returns:
            tuple: (states, actions, rewards, next_states, dones) where each is a list
                   of tensors, and dones is a single tensor.
        """
        experiences = random.sample(self.memory, self.batch_size)

        # Stack each field across the batch
        states = [torch.stack([e.states[i] for e in experiences]) for i in range(self.num_agents)]
        actions = [torch.stack([e.actions[i] for e in experiences]) for i in range(self.num_agents)]
        rewards = [torch.cat([e.rewards[i] for e in experiences]) for i in range(self.num_agents)]
        next_states = [torch.stack([e.next_states[i] for e in experiences]) for i in range(self.num_agents)]
        dones = torch.cat([e.dones for e in experiences])

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)