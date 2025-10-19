"""
video_recorder.py
------------------
Utility functions for recording episodes of Nina's environment.

Author: Lucio Nunes (Nina Project)
Date: 2025-10-19
"""

import os
import numpy as np
import pygame
import imageio.v3 as iio
from envs.cube_hunt_env import CubeHuntEnv


# ============================================================
# === Editable Parameters (easy to modify for experimentation)
# ============================================================

DEFAULT_OUTPUT_PATH = "nina_logs/videos/nina_episode"  # Base folder for video output
DEFAULT_FPS = 10                                       # Frames per second
GRID_SIZE = 10                                         # Grid size for environment


# ============================================================
# === Episode Recording Function
# ============================================================

def record_episode(model, filename_base: str = DEFAULT_OUTPUT_PATH, fps: int = DEFAULT_FPS):
    """
    Records a full episode of Nina chasing the rabbit.
    Automatically increments the output filename (v01, v02, ...).

    Args:
        model: trained PPO model (Stable Baselines3)
        filename_base (str): base path for the video file (no extension)
        fps (int): frames per second for the output video
    """

    # === Ensure output folder exists ===
    os.makedirs(os.path.dirname(filename_base), exist_ok=True)

    # === Find next available version ===
    version = 1
    while os.path.exists(f"{filename_base}_v{version:02d}.mp4"):
        version += 1
    filename = f"{filename_base}_v{version:02d}.mp4"

    # === Create environment ===
    env = CubeHuntEnv(grid_size=GRID_SIZE)
    obs, _ = env.reset()
    frames = []
    done = False

    print(f"🎬 Starting video recording ({filename}) ...")

    # === Run episode ===
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        env.render()

        # Capture frame from pygame surface
        frame = pygame.surfarray.array3d(env.screen)
        frame = np.transpose(frame, (1, 0, 2))  # Convert to (height, width, channels)
        frames.append(frame)

    # === Save video ===
    if frames:
        print(f"🎥 Recording video with {len(frames)} frames...")
        iio.imwrite(filename, frames, fps=fps, codec="libx264")
        print(f"✅ Video saved as {filename}")
    else:
        print("⚠️ No frames captured — ensure environment is rendering correctly.")

    env.close()
