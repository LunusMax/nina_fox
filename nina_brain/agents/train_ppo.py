from __future__ import annotations

"""
train_ppo.py
------------
Trains Nina's brain (PPO algorithm) in the CubeHunt environment.

Author: Lucio Nunes (Nina Project)
Date: 2025-10-20
"""

# ============================================================
# === Editable Parameters (easy to tweak)
# ============================================================

GRID_SIZE = 10              # Size of the environment grid
TOTAL_TIMESTEPS = 20_000    # Number of training steps
LEARNING_RATE = 3e-4        # PPO learning rate
N_STEPS = 256               # PPO rollout length
BATCH_SIZE = 64             # PPO batch size
GAMMA = 0.99                # Discount factor

RENDER_DURING_TEST = True   # If True, renders Nina after training
SAVE_VIDEO = True            # Record a single evaluation episode after training
SAVE_TRAINING_VIDEO = True   # Record a timelapse of the training (AI Warehouse style)
TRAIN_TIMELAPSE_INTERVAL = 10  # Capture one frame every N training steps
TRAIN_TIMELAPSE_FPS = 30       # FPS of the final timelapse video


# ============================================================
# === Imports
# ============================================================

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import os
import sys

# === Ensure project root (nina_fox) is visible to Python ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Environment and rendering
import gymnasium as gym
from gymnasium import spaces
import pygame

# RL algorithms
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Utilities
try:
    from utils.video_recorder import record_episode
except Exception:
    record_episode = None

try:
    from utils.training_recorder_callback import TrainingRecorderCallback
except Exception:
    TrainingRecorderCallback = None


# ============================================================
# === Grid Visualization Helpers
# ============================================================

CELL_SIZE = 50
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
BG_COLOR = (50, 120, 50)  # Background color (grass)
GRID_COLOR = (70, 70, 70) # Grid lines color


def draw_grid(surface):
    """Draws a simple grid on the pygame surface."""
    for x in range(0, WINDOW_SIZE + 1, CELL_SIZE):
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, WINDOW_SIZE), 1)
    for y in range(0, WINDOW_SIZE + 1, CELL_SIZE):
        pygame.draw.line(surface, GRID_COLOR, (0, y), (WINDOW_SIZE, y), 1)


def draw_cell(surface, col, row, color):
    """Draws a single cell at the given grid coordinates."""
    x = col * CELL_SIZE
    y = row * CELL_SIZE
    pygame.draw.rect(surface, color, (x, y, CELL_SIZE, CELL_SIZE))


# ============================================================
# === Color Setup (for Nina and Rabbit)
# ============================================================

NINA_COLOR = (255, 140, 0)
NINA_POS = [0, 0]
RABBIT_COLOR = (0, 0, 0)
RABBIT_POS = [GRID_SIZE - 1, GRID_SIZE - 1]


# ============================================================
# === Training Routine
# ============================================================

from envs.cube_hunt_env import CubeHuntEnv

if __name__ == "__main__":
    # === Create environment and monitor ===
    env = CubeHuntEnv(grid_size=GRID_SIZE)
    env = Monitor(env)  # Wrap environment to log episode stats

    # === Initialize PPO model ===
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
    )

    # === Optional: training timelapse callback (AI Warehouse style) ===
    callback = None
    if SAVE_TRAINING_VIDEO and TrainingRecorderCallback is not None:
        try:
            callback = TrainingRecorderCallback(
                frame_interval=TRAIN_TIMELAPSE_INTERVAL,
                fps=TRAIN_TIMELAPSE_FPS,
                output_path="nina_logs/videos/nina_training_timelapse",
                verbose=1,
            )
        except Exception as e:
            print(f"⚠️ Could not initialize training timelapse callback: {e}")
            callback = None

    # === Train Nina ===
    print("🚀 Training NINA (PPO)...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    # === Test trained agent ===
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        if RENDER_DURING_TEST:
            env.render()

    # === Save trained model ===
    model_dir = "nina_logs/models"
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "nina_ppo_model"))
    print("💾 Model saved in nina_logs/models/nina_ppo_model.zip")

    # === Record single evaluation episode (optional) ===
    if SAVE_VIDEO and record_episode is not None:
        video_dir = "nina_logs/videos"
        os.makedirs(video_dir, exist_ok=True)
        try:
            record_episode(model, filename_base=os.path.join(video_dir, "nina_learning"))
        except Exception as e:
            print(f"⚠️ Could not record video: {e}")

    print("✅ Training complete.")
