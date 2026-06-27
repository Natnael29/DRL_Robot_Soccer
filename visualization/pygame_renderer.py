# visualization/pygame_renderer.py
import pygame
import numpy as np
import sys

class SoccerRenderer:
    """Pygame-based renderer for the Robot Soccer environment."""
    
    def __init__(self, world_size=15.0, width=800, height=600):
        """
        Initialize the Pygame renderer.
        
        Args:
            world_size (float): Size of the soccer field (world_size x world_size).
            width (int): Screen width in pixels.
            height (int): Screen height in pixels.
        """
        pygame.init()
        self.width = width
        self.height = height
        self.world_size = world_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Robot Soccer League - MADDPG Training")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.bg_color = (34, 139, 34)      # Green field
        self.line_color = (255, 255, 255)  # White lines
        self.team0_color = (0, 0, 255)     # Blue team
        self.team1_color = (255, 0, 0)       # Red team
        self.ball_color = (255, 255, 0)    # Yellow ball
        self.goal_color = (255, 255, 255)  # White goals
        
        # Font for text
        self.font = pygame.font.Font(None, 36)
        
        # Calculate scale: world units to pixels
        self.scale = min(width, height) / (world_size * 1.2)
        self.offset_x = width // 2
        self.offset_y = height // 2
        
    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates."""
        screen_x = int(x * self.scale + self.offset_x)
        screen_y = int(self.height - (y * self.scale + self.offset_y))
        return screen_x, screen_y
    
    def render(self, agent_states, ball_state, scores=None):
        """
        Render the current state of the soccer environment.
        
        Args:
            agent_states (dict): Dictionary mapping agent_id to state dict.
            ball_state (dict): Dictionary with position and radius.
            scores (dict): Optional dict with team scores.
        """
        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        self.screen.fill(self.bg_color)
        
        # Draw field boundaries
        margin = 20
        pygame.draw.rect(self.screen, self.line_color, 
                        (margin, margin, self.width - 2*margin, self.height - 2*margin), 2)
        
        # Draw center line
        pygame.draw.line(self.screen, self.line_color, 
                        (self.width // 2, margin), (self.width // 2, self.height - margin), 2)
        
        # Draw goals
        goal_width_px = int(3.0 * self.scale)  # goal_width from config
        goal_height_px = int((self.world_size/2.5) * self.scale)
        
        # Left goal (Team 1 goal)
        goal_left_x = margin - 5
        goal_left_y = (self.height // 2) - goal_height_px // 2
        pygame.draw.rect(self.screen, self.goal_color, 
                        (goal_left_x, goal_left_y, 10, goal_height_px), 0)
        
        # Right goal (Team 0 goal)
        goal_right_x = self.width - margin - 5
        pygame.draw.rect(self.screen, self.goal_color, 
                        (goal_right_x, goal_left_y, 10, goal_height_px), 0)
        
        # Draw ball
        ball_pos = ball_state['position']
        ball_radius_px = int(ball_state['radius'] * self.scale * 5)  # Scale up for visibility
        ball_screen_x, ball_screen_y = self.world_to_screen(ball_pos[0], ball_pos[1])
        pygame.draw.circle(self.screen, self.ball_color, 
                          (ball_screen_x, ball_screen_y), max(ball_radius_px, 8))
        
        # Draw agents
        for agent_id, state in agent_states.items():
            pos = state['position']
            team = state['team']
            robot_radius_px = int(0.1 * self.scale * 5)  # Scale up for visibility
            
            # Team 0 = Blue, Team 1 = Red
            color = self.team0_color if team == 0 else self.team1_color
            
            screen_x, screen_y = self.world_to_screen(pos[0], pos[1])
            pygame.draw.circle(self.screen, color, 
                              (screen_x, screen_y), max(robot_radius_px, 12))
            
            # Draw agent ID (for debugging)
            text = self.font.render(agent_id[-1], True, (255, 255, 255))
            text_rect = text.get_rect(center=(screen_x, screen_y))
            self.screen.blit(text, text_rect)
        
        # Draw scores
        if scores:
            score_text = self.font.render(f"Team 0: {scores.get(0, 0)}  Team 1: {scores.get(1, 0)}", 
                                         True, (255, 255, 255))
            self.screen.blit(score_text, (10, 10))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS
    
    def close(self):
        """Clean up Pygame resources."""
        pygame.quit()

# Example usage
if __name__ == "__main__":
    renderer = SoccerRenderer()
    
    # Dummy test
    agent_states = {
        'agent_0': {'position': np.array([0, 0]), 'team': 0},
        'agent_1': {'position': np.array([2, 2]), 'team': 0},
    }
    ball_state = {
        'position': np.array([5, 5]),
        'radius': 0.1
    }
    
    running = True
    while running:
        renderer.render(agent_states, ball_state)
        import time
        time.sleep(0.1)  # Slow down for demo