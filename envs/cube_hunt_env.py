"""
cube_hunt_env.py
----------------
Custom environment for the Nina Project.

Author: Lucio Nunes (Nina Project)
Date: 2025-10-19

Description:
------------
Simple grid-based environment where Nina (the orange cube)
tries to catch the black cube (the rabbit).

Observation: [x_fox, y_fox, x_rabbit, y_rabbit]
Actions: 0=up, 1=right, 2=down, 3=left
"""

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

import os
import sys

# Ensure project root (nina_fox) is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ============================================================
# === Editable Parameters (easy to modify for experimentation)
# ============================================================

GRID_SIZE = 10            # Number of cells on each side of the grid
WINDOW_SIZE = 400         # Window size in pixels
BG_COLOR = (50, 120, 50)  # Background color (grass)
GRID_COLOR = (70, 70, 70) # Grid line color
FOX_COLOR = (255, 140, 0) # Nina's color
RABBIT_COLOR = (0, 0, 0)  # Rabbit's color
FPS = 5                   # Frames per second for rendering
RANDOM_START = False      # If True, Nina and rabbit start in random positions
STEP_PENALTY = -0.01      # Small penalty for each move
CATCH_REWARD = 1.0        # Reward for catching the rabbit


# ============================================================
# === Environment Definition
# ============================================================

class CubeHuntEnv(gym.Env):
    """Simple grid environment where Nina chases a rabbit."""
    
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, grid_size=GRID_SIZE, render_mode="human", random_start=RANDOM_START):
        """Initialize the environment."""
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.random_start = random_start

        # === Define Action and Observation Spaces ===
        self.action_space = spaces.Discrete(4)  # 0=up, 1=right, 2=down, 3=left
        low = np.array([0, 0, 0, 0], dtype=np.int32)
        high = np.array([grid_size - 1, grid_size - 1, grid_size - 1, grid_size - 1], dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # === Initialize Pygame ===
        pygame.init()
        self.window_size = WINDOW_SIZE
        self.cell_size = self.window_size // self.grid_size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("NINA - Cube Hunt")
        self.clock = pygame.time.Clock()

        # === Initial State ===
        self.fox_pos = np.array([0, 0])
        self.rabbit_pos = np.array([self.grid_size - 1, self.grid_size - 1])

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state."""
        super().reset(seed=seed)

        if self.random_start:
            # Random starting positions (cannot overlap)
            self.fox_pos = np.random.randint(0, self.grid_size, size=2)
            self.rabbit_pos = np.random.randint(0, self.grid_size, size=2)
            while np.array_equal(self.fox_pos, self.rabbit_pos):
                self.rabbit_pos = np.random.randint(0, self.grid_size, size=2)
        else:
            self.fox_pos = np.array([0, 0])
            self.rabbit_pos = np.array([self.grid_size - 1, self.grid_size - 1])

        obs = np.concatenate([self.fox_pos, self.rabbit_pos])
        return obs, {}

    def step(self, action):
        """Execute one action in the environment and return the new state."""
        
        # === Move Nina ===
        if action == 0:      # Up
            self.fox_pos[1] = max(0, self.fox_pos[1] - 1)
        elif action == 1:    # Right
            self.fox_pos[0] = min(self.grid_size - 1, self.fox_pos[0] + 1)
        elif action == 2:    # Down
            self.fox_pos[1] = min(self.grid_size - 1, self.fox_pos[1] + 1)
        elif action == 3:    # Left
            self.fox_pos[0] = max(0, self.fox_pos[0] - 1)

        # === Reward Calculation ===
        caught = np.array_equal(self.fox_pos, self.rabbit_pos)
        reward = CATCH_REWARD if caught else STEP_PENALTY
        obs = np.concatenate([self.fox_pos, self.rabbit_pos])
        done = bool(caught)

        return obs, reward, done, False, {}

    def render(self):
        """Render the current state of the environment."""
        self.screen.fill(BG_COLOR)

        # === Draw Grid ===
        for x in range(0, self.window_size + 1, self.cell_size):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, self.window_size), 1)
        for y in range(0, self.window_size + 1, self.cell_size):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (self.window_size, y), 1)

        # === Draw Rabbit ===
        pygame.draw.rect(
            self.screen,
            RABBIT_COLOR,
            (self.rabbit_pos[0] * self.cell_size,
             self.rabbit_pos[1] * self.cell_size,
             self.cell_size, self.cell_size)
        )

        # === Draw Nina ===
        pygame.draw.rect(
            self.screen,
            FOX_COLOR,
            (self.fox_pos[0] * self.cell_size,
             self.fox_pos[1] * self.cell_size,
             self.cell_size, self.cell_size)
        )

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """Close the environment and release pygame resources."""
        pygame.quit()
