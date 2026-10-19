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

from pathlib import Path
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

GRID_SIZE = 10               # Number of cells on each side of the grid
WINDOW_SIZE = 400            # Window size in pixels
BG_COLOR = (50, 120, 50)     # Background color (grass)
GRID_COLOR = (70, 70, 70)    # Grid line color
FOX_COLOR = (255, 140, 0)    # Nina's color
RABBIT_COLOR = (0, 0, 0)     # Rabbit's color
FPS = 5                      # Frames per second for rendering
RANDOM_START = False         # If True, Nina and rabbit start in random positions
RANDOM_RABBIT_START = False  # Randomize only rabbit's starting position
STEP_PENALTY = -0.01         # Small penalty for each move
CATCH_REWARD = 1.0           # Reward for catching the rabbit
MAX_EPISODE_STEPS = 100


# ============================================================
# === Environment Definition
# ============================================================

class CubeHuntEnv(gym.Env):
    """Simple grid environment where Nina chases a rabbit."""
    
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, grid_size=GRID_SIZE, render_mode="human", random_start=RANDOM_START, random_rabbit_start=RANDOM_RABBIT_START,):
        """Initialize the environment."""
        super().__init__()

        self.grid_size = grid_size
        self.render_mode = render_mode
        self.random_start = random_start
        self.random_rabbit_start = random_rabbit_start
        self.episode_steps = 0
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

        # === Asset Paths ===
        base_dir = Path(__file__).resolve().parents[1]
        self.assets_dir = base_dir / "assets" / "sprites"

        self.fox_sprite_path = self.assets_dir / "fox_top.png"
        self.rabbit_sprite_path = self.assets_dir / "rabbit_top.png"

        # === Load Sprites ===
        self.fox_sprite = self._load_sprite(
            self.fox_sprite_path,
            (self.cell_size, self.cell_size)
        )
        rabbit_size = max(1, int(self.cell_size * 0.65))

        self.rabbit_sprite = self._load_sprite(
            self.rabbit_sprite_path,
            (rabbit_size, rabbit_size)
        )
        

        # === Initial State ===
        self.fox_pos = np.array([0, 0])
        self.rabbit_pos = np.array([self.grid_size - 1, self.grid_size - 1])


    def _load_sprite(self, path, size):
        """Load and resize a sprite while preserving transparency."""
        sprite = pygame.image.load(str(path)).convert_alpha()
        sprite = pygame.transform.scale(sprite, size)

        return sprite


    def reset(self, seed=None, options=None):
        """Reset Nina and the rabbit to their starting positions."""
        super().reset(seed=seed)
        self.episode_steps = 0

        # === Nina's Starting Position ===
        if self.random_start:
            self.fox_pos = self.np_random.integers(
                0, self.grid_size, size=2, dtype=np.int32
            )
        else:
            self.fox_pos = np.array([0, 0], dtype=np.int32)

        # === Rabbit's Starting Position ===
        if self.random_start or self.random_rabbit_start:
            self.rabbit_pos = self.np_random.integers(
                0, self.grid_size, size=2, dtype=np.int32
            )

            # Nina and the rabbit cannot start in the same cell
            while np.array_equal(self.fox_pos, self.rabbit_pos):
                self.rabbit_pos = self.np_random.integers(
                    0, self.grid_size, size=2, dtype=np.int32
                )
        else:
            self.rabbit_pos = np.array(
                [self.grid_size - 1, self.grid_size - 1],
                dtype=np.int32
            )

        obs = np.concatenate([self.fox_pos, self.rabbit_pos])
        return obs, {}


    def step(self, action):
        """Execute one action and calculate distance-based rewards."""

        # === Distance Before Movement ===
        old_distance = np.abs(
            self.fox_pos - self.rabbit_pos
        ).sum()

        # === Store Position Before Movement ===
        old_pos = self.fox_pos.copy()

        # === Move Nina ===
        if action == 0:      # Up
            self.fox_pos[1] = max(0, self.fox_pos[1] - 1)

        elif action == 1:    # Right
            self.fox_pos[0] = min(
                self.grid_size - 1, self.fox_pos[0] + 1
            )

        elif action == 2:    # Down
            self.fox_pos[1] = min(
                self.grid_size - 1, self.fox_pos[1] + 1
            )

        elif action == 3:    # Left
            self.fox_pos[0] = max(0, self.fox_pos[0] - 1)

        # === Distance After Movement ===
        new_distance = np.abs(
            self.fox_pos - self.rabbit_pos
        ).sum()

        # === Reward Calculation ===
        caught = np.array_equal(
            self.fox_pos,
            self.rabbit_pos
        )

        blocked = np.array_equal(
            old_pos,
            self.fox_pos
        )

        if caught:
            reward = CATCH_REWARD

        elif blocked:
            reward = -0.05

        else:
            distance_improvement = old_distance - new_distance

            reward = (
                STEP_PENALTY
                + 0.02 * distance_improvement
            )
            
        # === Observation ===
        obs = np.concatenate([
            self.fox_pos,
            self.rabbit_pos
        ])

        self.episode_steps += 1

        terminated = bool(caught)
        truncated = (
            not terminated
            and self.episode_steps >= MAX_EPISODE_STEPS
        )

        return obs, reward, terminated, truncated, {}


    def render(self):
        """Render the current state of the environment."""
        self.screen.fill(BG_COLOR)

        # === Draw Grid ===
        for x in range(0, self.window_size + 1, self.cell_size):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, self.window_size), 1)

        for y in range(0, self.window_size + 1, self.cell_size):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (self.window_size, y), 1)

        # === Draw Rabbit (centered inside the cell) ===
        rabbit_pixel_x = self.rabbit_pos[0] * self.cell_size
        rabbit_pixel_y = self.rabbit_pos[1] * self.cell_size

        rabbit_rect = self.rabbit_sprite.get_rect()
        rabbit_rect.center = (
            rabbit_pixel_x + self.cell_size // 2,
            rabbit_pixel_y + self.cell_size // 2
        )

        self.screen.blit(self.rabbit_sprite, rabbit_rect)

        # === Draw Nina ===
        fox_pixel_x = self.fox_pos[0] * self.cell_size
        fox_pixel_y = self.fox_pos[1] * self.cell_size

        self.screen.blit(self.fox_sprite, (fox_pixel_x, fox_pixel_y))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])


    def close(self):
        """Close the environment and release pygame resources."""
        pygame.quit()
