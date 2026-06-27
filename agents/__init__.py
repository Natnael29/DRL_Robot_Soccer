# agents/__init__.py
from .maddpg_agent import MADDPG
from .mlp_networks import MLP

__all__ = ['MADDPG', 'MLP']