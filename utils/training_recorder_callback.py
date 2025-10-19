"""
training_recorder_callback.py
-----------------------------
Callback that records the environment during training
to produce a timelapse of Nina's learning process.

Author: Lucio Nunes (Nina Project)
Date: 2025-10-21
"""

import os
import numpy as np
import pygame
import imageio.v3 as iio
from stable_baselines3.common.callbacks import BaseCallback


# ============================================================
# === Editable Parameters (easy to modify for experimentation)
# ============================================================

FRAME_INTERVAL = 20        # Record a frame every N training steps
VIDEO_FPS = 30             # Frames per second of the final timelapse video
OUTPUT_PATH = "nina_logs/videos/nina_training_timelapse"  # Base output path for the video
SHOW_ITERATION_COUNTER = True  # Display current iteration number on the frame


# ============================================================
# === Training Recorder Callback Definition
# ============================================================

class TrainingRecorderCallback(BaseCallback):
    """
    Records frames during training to create a timelapse video.

    The callback captures a frame every 'frame_interval' steps,
    then compiles them into a single accelerated video at the end of training.
    """

    def __init__(self, frame_interval=FRAME_INTERVAL, fps=VIDEO_FPS, output_path=OUTPUT_PATH, verbose=1):
        super().__init__(verbose)
        self.frame_interval = frame_interval
        self.fps = fps
        self.output_path = output_path
        self.frames = []
        self.version = 1
        pygame.font.init()
        self.font = pygame.font.SysFont("consolas", 18)

    def _on_step(self) -> bool:
        """
        Called at every step during training.
        Captures frames at specified intervals for the timelapse.
        """
        # === Ensure training environment exists ===
        if self.training_env is None:
            return True

        # === Capture a frame every N steps ===
        if self.n_calls % self.frame_interval == 0:
            try:
                # === Access the real environment (unwrap the Monitor wrapper) ===
                env = self.training_env.envs[0]
                base_env = getattr(env, "env", env)
                while hasattr(base_env, "env"):
                    base_env = base_env.env

                # === Render environment ===
                base_env.render()

                # === Overlay iteration count (optional) ===
                if SHOW_ITERATION_COUNTER:
                    iteration_text = f"Iteration: {self.n_calls}"
                    text_surface = self.font.render(iteration_text, True, (255, 255, 255))
                    base_env.screen.blit(text_surface, (10, 10))

                # === Capture frame ===
                frame = pygame.surfarray.array3d(base_env.screen)
                frame = np.transpose(frame, (1, 0, 2))
                self.frames.append(frame)

            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Frame capture failed at step {self.n_calls}: {e}")

        return True

    def _on_training_end(self) -> None:
        """
        Compiles all recorded frames into a single timelapse video
        at the end of training.
        """
        # === Ensure output directory exists ===
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # === Determine next available version ===
        while os.path.exists(f"{self.output_path}_v{self.version:02d}.mp4"):
            self.version += 1
        filename = f"{self.output_path}_v{self.version:02d}.mp4"

        # === Save the timelapse video ===
        if self.frames:
            print(f"🎞️ Saving training timelapse ({len(self.frames)} frames)...")
            try:
                iio.imwrite(filename, self.frames, fps=self.fps, codec="libx264")
                print(f"✅ Training timelapse saved as {filename}")
            except Exception as e:
                print(f"⚠️ Could not write timelapse video: {e}")
        else:
            print("⚠️ No frames recorded during training.")
