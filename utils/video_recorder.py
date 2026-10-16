"""
video_recorder.py
------------------
Utility functions for recording episodes of Nina's environment.

Author: Lucio Nunes (Nina Project)
Date: 2025-10-19
"""

from pathlib import Path

import numpy as np
import pygame
import imageio.v3 as iio

from envs.cube_hunt_env import CubeHuntEnv


# ============================================================
# === Editable Parameters
# ============================================================

DEFAULT_OUTPUT_PATH = "nina_logs/videos/nina_episode"
DEFAULT_FPS = 5
GRID_SIZE = 10
MAX_STEPS = 500

# ============================================================
# === Episode Recording Function
# ============================================================

def record_episode(
    model,
    filename_base: str = DEFAULT_OUTPUT_PATH,
    fps: int = DEFAULT_FPS,
    max_steps: int = MAX_STEPS,
    random_start: bool = False,
):
    """
    Records one episode of Nina chasing the rabbit.

    Automatically increments the output filename
    (v01, v02, ...).
    """

    # === Prepare output path ===
    base_path = Path(filename_base)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    version = 1

    while True:
        filename = base_path.parent / (
            f"{base_path.name}_v{version:02d}.mp4"
        )

        if not filename.exists():
            break

        version += 1

    # === Create environment ===
    env = CubeHuntEnv(
        grid_size=GRID_SIZE,
        random_start=random_start,
    )

    frames = []
    obs, _ = env.reset()

    terminated = False
    truncated = False

    print(f"🎬 Starting video recording ({filename}) ...")

    try:
        for step in range(max_steps):

            # === Handle window events ===
            if any(
                event.type == pygame.QUIT
                for event in pygame.event.get()
            ):
                print("⚠️ Recording interrupted.")
                break

            # === Predict action ===
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            # === Update environment ===
            obs, reward, terminated, truncated, _ = env.step(
                action
            )

            # === Render and capture frame ===
            env.render()

            frame = pygame.surfarray.array3d(env.screen)
            frame = np.transpose(frame, (1, 0, 2))
            frames.append(frame)

            # === Stop episode ===
            if terminated or truncated:
                break

        if terminated:
            print("🦊 Nina caught the rabbit!")

        elif truncated:
            print("⏱️ Episode truncated by environment.")

        else:
            print("⏱️ Recording ended before capture.")

    finally:
        env.close()

    # === Save video ===
    if frames:
        print(f"🎥 Recording {len(frames)} frames...")

        iio.imwrite(
            str(filename),
            np.stack(frames),
            fps=fps,
            codec="libx264",
        )

        print(f"✅ Video saved as {filename}")

    else:
        print("⚠️ No frames captured.")
