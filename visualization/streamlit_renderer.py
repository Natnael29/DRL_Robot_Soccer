# visualization/streamlit_renderer.py
"""In-memory renderer for Streamlit - generates images without display."""

import numpy as np
from PIL import Image, ImageDraw

class StreamlitRenderer:
    """Image-based renderer for Streamlit dashboard."""

    def __init__(self, world_size=15.0, width=600, height=450):
        """
        Initialize the renderer.

        Args:
            world_size (float): Size of the soccer field (world_size x world_size).
            width (int): Image width in pixels.
            height (int): Image height in pixels.
        """
        self.width = width
        self.height = height
        self.world_size = world_size

        # Colors (RGB)
        self.bg_color = (34, 139, 34)       # Green field
        self.line_color = (255, 255, 255)   # White lines
        self.team0_color = (0, 0, 255)      # Blue team
        self.team1_color = (255, 0, 0)       # Red team
        self.ball_color = (255, 255, 0)     # Yellow ball
        self.goal_color = (200, 200, 200)   # Gray goals

        # Calculate scale: world units to pixels
        self.scale = min(width, height) / (world_size * 1.1)
        self.offset_x = width // 2
        self.offset_y = height // 2

    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates."""
        screen_x = int(x * self.scale + self.offset_x)
        screen_y = int(self.height - (y * self.scale + self.offset_y))
        return screen_x, screen_y

    def render(self, agent_states, ball_state, scores=None):
        """
        Render the current state and return as PIL Image.

        Args:
            agent_states (dict): Dictionary mapping agent_id to state dict.
            ball_state (dict): Dictionary with position and radius.
            scores (dict): Optional dict with team scores.

        Returns:
            PIL Image: The rendered frame.
        """
        # Create image
        img = Image.new('RGB', (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)

        margin = 20

        # Draw field boundary
        draw.rectangle([margin, margin, self.width - margin, self.height - margin],
                       outline=self.line_color, width=2)

        # Draw center line
        draw.line([self.width // 2, margin, self.width // 2, self.height - margin],
                  fill=self.line_color, width=2)

        # Draw center circle
        center_x, center_y = self.world_to_screen(0, 0)
        draw.ellipse([center_x - 30, center_y - 30, center_x + 30, center_y + 30],
                     outline=self.line_color, width=2)

        # Draw goals
        goal_height_px = int((self.world_size / 2.5) * self.scale * 0.6)

        # Left goal (Team 1's goal - team 0 scores here)
        goal_left_y = (self.height // 2) - goal_height_px // 2
        draw.rectangle([margin - 5, goal_left_y, margin + 5, goal_left_y + goal_height_px],
                       fill=self.goal_color)

        # Right goal (Team 0's goal - team 1 scores here)
        draw.rectangle([self.width - margin - 5, goal_left_y, self.width - margin + 5, goal_left_y + goal_height_px],
                       fill=self.goal_color)

        # Draw ball
        ball_pos = ball_state['position']
        ball_radius_px = max(int(ball_state['radius'] * self.scale * 3), 6)
        ball_screen_x, ball_screen_y = self.world_to_screen(ball_pos[0], ball_pos[1])
        draw.ellipse([ball_screen_x - ball_radius_px, ball_screen_y - ball_radius_px,
                      ball_screen_x + ball_radius_px, ball_screen_y + ball_radius_px],
                     fill=self.ball_color, outline=(200, 200, 0), width=1)

        # Draw agents
        for agent_id, state in agent_states.items():
            pos = state['position']
            team = state['team']
            robot_radius_px = max(int(0.1 * self.scale * 3), 8)

            # Team 0 = Blue, Team 1 = Red
            color = self.team0_color if team == 0 else self.team1_color

            screen_x, screen_y = self.world_to_screen(pos[0], pos[1])
            draw.ellipse([screen_x - robot_radius_px, screen_y - robot_radius_px,
                          screen_x + robot_radius_px, screen_y + robot_radius_px],
                         fill=color, outline=(255, 255, 255), width=2)

            # Draw direction indicator
            orientation = state.get('orientation', 0)
            dir_x = screen_x + robot_radius_px * 0.8 * np.cos(orientation)
            dir_y = screen_y - robot_radius_px * 0.8 * np.sin(orientation)
            draw.line([screen_x, screen_y, dir_x, dir_y], fill=(255, 255, 255), width=2)

        # Draw scores
        if scores:
            score_text = f"Blue (Team 0): {scores.get(0, 0)}   Red (Team 1): {scores.get(1, 0)}"
            draw.text((10, 10), score_text, fill=(255, 255, 255))

        return img

    def close(self):
        """No cleanup needed for PIL-based renderer."""
        pass


def render_soccer_animation(env, maddpg_agent=None, num_steps=100, fps=15, width=600, height=450):
    """
    Run the environment and render animation frames.

    Args:
        env: RobotSoccerEnv instance
        maddpg_agent: Optional MADDPG agent to control actions. If None, random actions are used.
        num_steps: Number of steps to simulate
        fps: Frames per second for the animation
        width: Frame width
        height: Frame height

    Returns:
        List of PIL Images (frames)
    """
    renderer = StreamlitRenderer(world_size=env.world_size, width=width, height=height)
    frames = []
    scores = {0: 0, 1: 0}

    # Reset environment
    observations, _ = env.reset()

    for step in range(num_steps):
        if maddpg_agent:
            # Agent selects actions
            actions = maddpg_agent.select_actions(observations, explore=False)
        else:
            # Random actions
            actions = {}
            for agent in env.agents:
                actions[agent] = {
                    'linear_vel': np.random.uniform(-0.5, 0.5, 1).astype(np.float32),
                    'angular_vel': np.random.uniform(-1, 1, 1).astype(np.float32),
                    'kick': np.random.randint(0, 2)
                }

        # Step the environment
        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        # Check for goals
        for agent, reward in rewards.items():
            team = env.agent_states[agent]['team']
            if reward > 5:  # Goal scored
                scores[team] = scores.get(team, 0) + 1

        # Render frame
        frame = renderer.render(env.agent_states, env.ball_state, scores)
        frames.append(frame)

        if any(terminations.values()) or any(truncations.values()):
            break

        # Update observations for next step
        observations = next_observations

    renderer.close()
    return frames