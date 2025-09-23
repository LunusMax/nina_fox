# Import modules
from __future__ import annotations
import math
import random # Generate random positions for rabbit
import time
from dataclasses import dataclass
from pathlib import Path # Save archives and logs
from typing import Optional, Tuple, List
import numpy as np

# Import ambience
import gymnasium as gym
from gymnasium import spaces
import pygame


# Import RL algorithms
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback # Stop early
from stable_baselines3.common.monitor import Monitor # Metrics wrapper

# Import record
try:
    import imageio.v3 as iio # Save frame sequence
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

# Create grid
GRID_SIZE = 10
CELL_SIZE = 50
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
BG_COLOR = (50, 120, 50) # Green as grass
GRID_COLOR = (70, 70, 70)

# Function to draw grid
def draw_grid(surface):
    # vertical lines
    for x in range(0, WINDOW_SIZE + 1, CELL_SIZE):
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, WINDOW_SIZE), 1)
    # horizontal lines
    for y in range(0, WINDOW_SIZE + 1, CELL_SIZE):
        pygame.draw.line(surface, GRID_COLOR, (0, y), (WINDOW_SIZE, y), 1)

# Function to draw cell

def draw_cell(surface, col, row, color):
    x = col * CELL_SIZE
    y = row * CELL_SIZE
    pygame.draw.rect(surface, color, (x, y, CELL_SIZE, CELL_SIZE))

# Create cubes
NINA_COLOR = (255, 140, 0)
NINA_POS = [0, 0]
RABBIT_COLOR = (0, 0, 0)
RABBIT_POS = [GRID_SIZE - 1, GRID_SIZE - 1]

# Def main game
from cube_hunt_env import CubeHuntEnv

if __name__ == "__main__":
    # Create enviroment with Monitor
    env = CubeHuntEnv(grid_size=10)
    env = Monitor(env) # Collect metrics

    # PPO model
    model = PPO(
        policy="MlpPolicy",    # rede neural simples
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
    )

    # Train Nina
    print("🚀 Training NINA...")
    model.learn(total_timesteps=20_000)  # number of steps

    # Test the model
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        env.render()

    record_episode(model, "nina_learning_v01.mp4", fps=10)


# Record what Nina is doing
def record_episode(model, filename="nina_episode_v01.mp4", fps=10):
    env = CubeHuntEnv(grid_size=10)
    obs, _ = env.reset()
    frames = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        # render e capture
        env.render()
        frame = pygame.surfarray.array3d(env.screen)
        frame = np.transpose(frame, (1, 0, 2))        
        frames.append(frame)

    # salva o vídeo
    print(f"🎥 Recording video com {len(frames)} frames...")
    iio.imwrite(filename, frames, fps=fps, codec="libx264")
    print(f"✅ Video saved em {filename}")

