# robot_soccer_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo.utils import ParallelEnv

# --- Constants ---
NUM_AGENTS = 10 # 5 teammates + 5 opponents
TEAM_SIZE = 5
BALL_RADIUS = 0.1
ROBOT_RADIUS = 0.1

# Define state dimensions
AGENT_STATE_DIM = 3 # position (x, y) + orientation (radians)
VELOCITY_DIM = 2    # velocity (vx, vy)
BALL_STATE_DIM = 2  # position (x, y)
BALL_VELOCITY_DIM = 2 # velocity (vx, vy)

# Total observation dimensions per agent
# Each agent's observation vector will be structured as:
# [self_x, self_y, self_orientation, self_vx, self_vy,  # Self state (5 dims)
#  ball_x, ball_y, ball_vx, ball_vy,                   # Ball state (4 dims)
#  opp1_x, opp1_y, opp1_ori, opp1_vx, opp1_vy, ..., opp5_x, ..., opp5_vy, # Opponent states (5 agents * 5 dims = 25 dims)
#  team1_x, team1_y, team1_ori, team1_vx, team1_vy, ..., team4_x, ..., team4_vy] # Teammate states (4 agents * 5 dims = 20 dims)
# Total = 5 (self + vel) + 4 (ball + vel) + 25 (opponents) + 20 (teammates) = 54 dimensions
OBS_DIM_REFINED = (
    AGENT_STATE_DIM + VELOCITY_DIM # self state + velocity
    + BALL_STATE_DIM + BALL_VELOCITY_DIM # ball state + velocity
    + (NUM_AGENTS - 1) * (AGENT_STATE_DIM + VELOCITY_DIM) # other agents (self state + velocity)
)

# Define action dimensions
# Continuous actions: linear_velocity (forward/backward), angular_velocity (turning)
# These are defined as shape=(1,) for each in action_spaces for clarity.

# Discrete action: kick (0: no kick, 1: kick)
DISCRETE_ACTION_N = 2 # Number of discrete actions for kick

# --- Environment Class ---

class RobotSoccerEnv(ParallelEnv):
    """
    A custom PettingZoo environment for Robot Soccer with MADDPG.
    Agents observe their own state, ball state, and all other agents' states.
    Actions include continuous movement and a discrete kick.
    """
    metadata = {
        "render_modes": ["human"], # Add more render modes if needed
        "name": "robot_soccer_v0",
    }

    def __init__(self, max_cycles=1000):
        super().__init__()

        self.team_size = TEAM_SIZE
        self.max_cycles = max_cycles
        
        # Initialize agents list first using the constant
        self.agents = [f"agent_{i}" for i in range(NUM_AGENTS)]
        # REMOVED: Direct assignment to self.num_agents.
        # We can use len(self.agents) whenever num_agents is strictly needed.
        
        self.agent_selection = None # Used for rendering/sequential logic if needed
        self.steps = 0

        # Define observation space for each agent (flattened vector)
        self.observation_spaces = {agent: spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM_REFINED,), dtype=np.float32) for agent in self.agents}

        # Define action space for each agent
        # Dict space: components for continuous movement and discrete kick
        self.action_spaces = {agent: spaces.Dict({
            'linear_vel': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32), # Forward/backward speed
            'angular_vel': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32), # Turning speed
            'kick': spaces.Discrete(DISCRETE_ACTION_N) # 0: no kick, 1: kick
        }) for agent in self.agents}

        # Internal environment state (to be initialized in reset)
        self.world_size = 15.0 # Example world size
        self.agent_states = {} # Store states for each agent
        self.ball_state = {}   # Store ball state

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        if seed is not None:
            np.random.seed(seed)

        self.steps = 0
        # Ensure agents are re-initialized on reset if needed, though `self.agents` is constant
        self.agents = [f"agent_{i}" for i in range(NUM_AGENTS)] # Ensure consistency
        self.agent_selection = self.agents[0] # Start with agent_0 for rendering/sequential logic

        # Initialize agent positions, orientations, and velocities
        for i, agent_id in enumerate(self.agents):
            # Assign agents to teams (e.g., first 5 to team 0, next 5 to team 1)
            team_id = 0 if i < self.team_size else 1
            # Spread agents within their half of the field
            side_multiplier = 1 if team_id == 0 else -1
            pos_x = np.random.uniform(side_multiplier * self.world_size/3.0, side_multiplier * self.world_size/2.0)
            pos_y = np.random.uniform(-self.world_size/2.5, self.world_size/2.5)
            orientation = np.random.uniform(-np.pi, np.pi)

            self.agent_states[agent_id] = {
                'position': np.array([pos_x, pos_y], dtype=np.float32),
                'orientation': orientation,
                'velocity': np.array([0.0, 0.0], dtype=np.float32),
                'radius': ROBOT_RADIUS,
                'team': team_id
            }

        # Initialize ball position and velocity
        self.ball_state = {
            'position': np.array([0.0, 0.0], dtype=np.float32), # Start ball at center
            'velocity': np.array([0.0, 0.0], dtype=np.float32),
            'radius': BALL_RADIUS
        }

        # PettingZoo requires returning observations and info
        observations = {}
        for agent in self.agents:
            observations[agent] = self._get_observation(agent)
        infos = {a: {} for a in self.agents} # Placeholder for agent-specific info

        return observations, infos

    def _get_observation(self, agent_id):
        """Generates the observation vector for a given agent."""
        my_state = self.agent_states[agent_id]
        my_obs = np.concatenate([
            my_state['position'],
            [my_state['orientation']], # scalar orientation
            my_state['velocity']
        ]) # Shape: (2+1+2) = 5

        ball_obs = np.concatenate([
            self.ball_state['position'],
            self.ball_state['velocity']
        ]) # Shape: (2+2) = 4

        other_agents_obs_list = []
        for other_agent_id in self.agents:
            if other_agent_id != agent_id:
                other_state = self.agent_states[other_agent_id]
                other_agent_obs = np.concatenate([
                    other_state['position'],
                    [other_state['orientation']], # scalar orientation
                    other_state['velocity']
                ]) # Shape: (2+1+2) = 5
                other_agents_obs_list.append(other_agent_obs)

        # Flatten the list of other agents' observations
        # This will be a np array of shape (9 * 5,) = (45,)
        other_agents_obs_flat = np.array(other_agents_obs_list).flatten()

        # Combine all parts according to OBS_DIM_REFINED
        full_observation = np.concatenate([my_obs, ball_obs, other_agents_obs_flat])

        # Ensure the shape matches OBS_DIM_REFINED
        if full_observation.shape[0] != OBS_DIM_REFINED:
            print(f"Warning: Observation dimension mismatch for {agent_id}. Expected {OBS_DIM_REFINED}, got {full_observation.shape[0]}. "
                  "This might indicate an error in observation construction or constant definitions.")
            # Attempt to pad or truncate if dimensions don't match exactly
            if full_observation.shape[0] > OBS_DIM_REFINED:
                full_observation = full_observation[:OBS_DIM_REFINED]
            else:
                full_observation = np.pad(full_observation, (0, OBS_DIM_REFINED - full_observation.shape[0]), 'constant')
        
        return full_observation.astype(np.float32)

    def _apply_action(self, agent_id, action, dt=0.05): # dt is simulation time step
        """Applies an agent's action to update its state with simplified physics."""
        state = self.agent_states[agent_id]
        linear_vel_forward = action['linear_vel'][0] # Scalar from the Box space
        angular_vel = action['angular_vel'][0]       # Scalar from the Box space
        kick_action = action['kick']                 # Discrete kick action

        # --- Update Orientation ---
        state['orientation'] += angular_vel * dt
        # Normalize orientation to be within [-pi, pi]
        state['orientation'] = np.arctan2(np.sin(state['orientation']), np.cos(state['orientation']))

        # --- Update Position based on linear velocity and orientation ---
        # Calculate movement direction based on current orientation
        direction_x = np.cos(state['orientation'])
        direction_y = np.sin(state['orientation'])
        
        # Calculate movement vector. Assuming linear_vel is forward/backward speed.
        # We use linear_vel_forward[0] as the forward/backward speed magnitude.
        move_x = direction_x * linear_vel_forward
        move_y = direction_y * linear_vel_forward

        # Update position
        state['position'][0] += move_x * dt
        state['position'][1] += move_y * dt

        # --- Update Velocity (simplified) ---
        # Set current velocity based on movement direction and speed.
        # In a real physics engine, velocity would update due to forces.
        # This is an approximation: velocity = displacement / time_step
        state['velocity'] = np.array([move_x/dt, move_y/dt], dtype=np.float32)

        # --- Apply World Bounds ---
        half_world = self.world_size / 2.0
        state['position'][0] = np.clip(state['position'][0], -half_world, half_world)
        state['position'][1] = np.clip(state['position'][1], -half_world, half_world)

        # --- Kick Logic ---
        if kick_action == 1:
            # Placeholder for kick logic: applies force to the ball
            # This would involve calculating the impulse applied to the ball
            # based on the agent's orientation, kick strength, and mass.
            pass # Kick logic will be implemented later if needed

        return state # Return updated state

    def _apply_ball_physics(self, dt=0.05):
        """Applies physics to the ball. Placeholder for now."""
        # This is a highly simplified placeholder. A real physics simulation
        # would require handling collisions with world boundaries, goals, and agents,
        # as well as ball friction, bouncing, etc.

        # Example: If the ball is stationary at the center for too long, introduce slight randomness
        if np.allclose(self.ball_state['velocity'], 0, atol=1e-4) and np.allclose(self.ball_state['position'], [0,0], atol=1e-4):
             if np.random.rand() < 0.001: # Very small chance to add slight drift
                 self.ball_state['velocity'] = np.random.randn(2) * 0.1

        # Apply physics to ball movement
        self.ball_state['position'] += self.ball_state['velocity'] * dt
        self.ball_state['velocity'] *= 0.99 # Simple friction/drag

        # --- World Bounds for Ball ---
        half_world = self.world_size / 2.0
        ball_pos = self.ball_state['position']
        ball_vel = self.ball_state['velocity']
        ball_rad = self.ball_state['radius']

        # Bounce off walls (simplified, ignoring agent collisions for now)
        if ball_pos[0] - ball_rad < -half_world:
            self.ball_state['position'][0] = -half_world + ball_rad
            self.ball_state['velocity'][0] *= -0.9 # Bounce with damping
        if ball_pos[0] + ball_rad > half_world:
            self.ball_state['position'][0] = half_world - ball_rad
            self.ball_state['velocity'][0] *= -0.9 # Bounce with damping
        if ball_pos[1] - ball_rad < -half_world:
            self.ball_state['position'][1] = -half_world + ball_rad
            self.ball_state['velocity'][1] *= -0.9 # Bounce with damping
        if ball_pos[1] + ball_rad > half_world:
            self.ball_state['position'][1] = half_world - ball_rad
            self.ball_state['velocity'][1] *= -0.9 # Bounce with damping

        # --- Goal Logic (Placeholder) ---
        # Placeholder for checking if ball is in goal, agents scoring etc.
        # This would involve checking ball_state['position'] against goal areas.
        # For now, goals are not simulated.

    def step(self, actions):
        """
        Takes a step in the environment.
        actions: A dictionary mapping agent_id to their action.
        """
        if self.agent_selection is None:
            raise Exception("Call reset() before step().")

        # Apply actions to all agents
        for agent_id, action in actions.items():
            self._apply_action(agent_id, action)

        # Apply physics to the ball
        self._apply_ball_physics()

        # Game logic updates: check for goals, collisions, score, etc.
        # This is a placeholder area for game rules.
        # For example, determining if a goal was scored, updating scores, etc.

        self.steps += 1

        # Check if episode is done
        terminations = {agent_id: False for agent_id in self.agents}
        truncations = {agent_id: False for agent_id in self.agents}

        if self.steps >= self.max_cycles:
            # Truncate if max cycles reached (episode ends due to time limit)
            for agent_id in self.agents:
                truncations[agent_id] = True
        
        # Add termination logic for actual game conditions like scoring a goal.
        # e.g., if ball_in_goal: terminations = {agent_id: True for agent_id in self.agents}

        # Prepare rewards (PLACEHOLDER: All rewards are 0 for now)
        # Reward structure will be crucial for MADDPG to learn strategies.
        # Example: +1 for scoring, -1 for conceding, small reward for passing, penalty for errors.
        rewards = {agent_id: 0.0 for agent_id in self.agents}

        # Get observations for the next step
        observations = {}
        for agent in self.agents:
            observations[agent] = self._get_observation(agent)

        # Info dictionary for auxiliary information
        infos = {agent_id: {} for agent_id in self.agents}

        # Update agent_selection for rendering/sequential logic if needed
        # For ParallelEnv, all agents act simultaneously, so agent_selection logic is less critical for step flow
        # but needs to be managed if, for example, rendering is sequential.
        self.agent_selection = self.agents[(self.agents.index(self.agent_selection) + 1) % len(self.agents)]

        # PettingZoo's ParallelEnv step returns all observations, rewards, terminations, truncations, infos at once.
        return observations, rewards, terminations, truncations, infos

    def render(self, mode='human'):
        """Renders the environment."""
        if mode == 'human':
            # Basic text-based rendering focusing on key states
            print(f"--- Step: {self.steps} ---")
            print(f"World Size: {self.world_size}x{self.world_size}")
            print(f"Ball: Pos={self.ball_state['position']}, Vel={self.ball_state['velocity']}")
            
            # Display agent info, perhaps grouped by team if teams are determined
            team_0_agents = [agent for agent in self.agents if self.agent_states[agent]['team'] == 0]
            team_1_agents = [agent for agent in self.agents if self.agent_states[agent]['team'] == 1]

            print("Team 0 Agents:")
            for agent_id in team_0_agents:
                state = self.agent_states[agent_id]
                # Use degrees for orientation in render output for better readability
                print(f"  - {agent_id}: Pos={state['position']}, Ori={np.degrees(state['orientation']):.1f}°, Vel={state['velocity']}")
            
            print("Team 1 Agents:")
            for agent_id in team_1_agents:
                state = self.agent_states[agent_id]
                print(f"  - {agent_id}: Pos={state['position']}, Ori={np.degrees(state['orientation']):.1f}°, Vel={state['velocity']}")
        else:
            super().render(mode=mode) # Delegate to parent for other render modes if implemented

    # PettingZoo ParallelEnv requires these methods to be defined, even if they just return the same space for all agents.
    def observation_space(self, agent_id):
        """Returns the observation space for the specified agent. In ParallelEnv, it's usually the same for all."""
        return self.observation_spaces[agent_id]

    def action_space(self, agent_id):
        """Returns the action space for the specified agent. In ParallelEnv, it's usually the same for all."""
        return self.action_spaces[agent_id]

    # Note: There's no need for `_agent_selector` in a pure ParallelEnv step,
    # as all agents act simultaneously. It's mainly for rendering or sequential envs.
