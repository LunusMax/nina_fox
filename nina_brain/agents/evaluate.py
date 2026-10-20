
"""
evaluate.py
-----------
Evaluates a trained PPO agent across every possible
rabbit position in the CubeHunt environment.

Author: Lucio Nunes (Nina Project)
"""

import argparse
import csv
import ast
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from envs.cube_hunt_env import CubeHuntEnv


# ============================================================
# === Editable Parameters
# ============================================================

GRID_SIZE = 10
MAX_EVALUATION_STEPS = 100

MODEL_DIR = Path("nina_logs/models")
OUTPUT_DIR = Path("nina_logs/evaluations")

# ============================================================
# === Experiment Version
# ============================================================

def get_current_version():
    """Read the current experiment version from train_ppo.py."""

    train_script = (
        Path(__file__).resolve().parent
        / "agents"
        / "train_ppo.py"
    )

    source = train_script.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "VERSION":
                    version = ast.literal_eval(node.value)

                    if not isinstance(version, str):
                        raise ValueError("VERSION must be a string.")

                    return version

    raise ValueError(
        f"VERSION not found in {train_script}"
    )

# ============================================================
# === Model Loading
# ============================================================

def load_model(version):
    """Load a versioned PPO model, with or without .zip extension."""

    base_path = MODEL_DIR / f"nina_ppo_model_{version}"

    candidates = [
        Path(f"{base_path}.zip"),
        base_path,
    ]

    for path in candidates:
        if path.is_file():
            print(f"Loading model: {path}")
            return PPO.load(str(path), device="cpu")

    raise FileNotFoundError(
        f"No model found for version {version}"
    )


# ============================================================
# === Evaluation
# ============================================================

def evaluate(model, grid_size=GRID_SIZE):
    """Evaluate all possible rabbit positions."""

    env = CubeHuntEnv(
        grid_size=grid_size,
        render_mode=None,
        random_start=False,
        random_rabbit_start=False,
    )

    results = []

    try:
        for y in range(grid_size):
            for x in range(grid_size):

                if (x, y) == (0, 0):
                    continue

                obs, _ = env.reset()

                # Set a deterministic evaluation target
                env.rabbit_pos = np.array(
                    [x, y], dtype=np.int32
                )

                obs = np.concatenate([
                    env.fox_pos,
                    env.rabbit_pos,
                ])

                blocked_moves = 0
                success = False

                for step in range(1, MAX_EVALUATION_STEPS + 1):
                    previous_pos = env.fox_pos.copy()

                    action, _ = model.predict(
                        obs,
                        deterministic=True,
                    )

                    obs, reward, terminated, truncated, _ = env.step(
                        int(action)
                    )

                    if np.array_equal(previous_pos, env.fox_pos):
                        blocked_moves += 1

                    if terminated:
                        success = True

                    if terminated or truncated:
                        break

                # Optimal distance for an obstacle-free grid
                optimal_steps = x + y

                results.append({
                    "rabbit_x": x,
                    "rabbit_y": y,
                    "success": success,
                    "steps": step,
                    "optimal_steps": optimal_steps,
                    "extra_steps": (
                        step - optimal_steps if success else ""
                    ),
                    "blocked_moves": blocked_moves,
                })

    finally:
        env.close()

    return results


# ============================================================
# === Results
# ============================================================

def save_results(version, results):
    """Export detailed evaluation results to CSV."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"evaluation_{version}.csv"

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    return output_path


def print_summary(version, results):
    """Display aggregate evaluation metrics."""

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    total = len(results)
    success_rate = 100 * len(successful) / total

    mean_steps = (
        np.mean([r["steps"] for r in successful])
        if successful else float("nan")
    )

    mean_extra_steps = (
        np.mean([r["extra_steps"] for r in successful])
        if successful else float("nan")
    )

    total_blocked = sum(
        r["blocked_moves"] for r in results
    )

    optimal_captures = sum(
        r["extra_steps"] == 0 for r in successful
    )

    print("\n========== NINA EVALUATION ==========")
    print(f"Version: {version}")
    print(f"Evaluated positions: {total}")
    print(f"Successful captures: {len(successful)}")
    print(f"Failed captures: {len(failed)}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Mean capture steps: {mean_steps:.2f}")
    print(f"Mean extra steps: {mean_extra_steps:.2f}")
    print(f"Optimal-path captures: {optimal_captures}")
    print(f"Total blocked moves: {total_blocked}")

    if failed:
        print("\nFailed rabbit positions:")
        for result in failed:
            print(
                f"  ({result['rabbit_x']}, "
                f"{result['rabbit_y']})"
            )


# ============================================================
# === Entry Point
# ============================================================

def main():
    version = get_current_version()

    print(f"🦊 Current experiment version: {version}")

    model = load_model(version)
    results = evaluate(model)

    print_summary(version, results)

    output_path = save_results(version, results)
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()