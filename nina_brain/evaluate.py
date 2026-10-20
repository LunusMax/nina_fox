
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

from collections import deque

from envs.cube_hunt_env import CubeHuntEnv

from utils.report_generator import generate_report


# ============================================================
# === Editable Parameters
# ============================================================

GRID_SIZE = 10
MAX_EVALUATION_STEPS = 100

MODEL_DIR = Path("nina_logs/models")
OUTPUT_DIR = Path("nina_logs/evaluations")

LAKE_EVALUATION_SEEDS = range(42, 52)  # 10 reproducible lakes

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


def shortest_path_distances(grid_size, obstacles):
    """Calculate shortest distances from (0, 0) using BFS."""

    start = (0, 0)
    distances = {start: 0}
    queue = deque([start])

    directions = [
        (0, -1),
        (1, 0),
        (0, 1),
        (-1, 0),
    ]

    while queue:
        x, y = queue.popleft()

        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            nx, ny = neighbor

            if not (0 <= nx < grid_size and 0 <= ny < grid_size):
                continue

            if neighbor in obstacles or neighbor in distances:
                continue

            distances[neighbor] = distances[(x, y)] + 1
            queue.append(neighbor)

    return distances

# ============================================================
# === Evaluation
# ============================================================

def evaluate(model, grid_size=GRID_SIZE):
    """Evaluate the agent across valid environments and targets."""

    # === Detect Environment Type ===
    observation_size = model.observation_space.shape[0]

    if observation_size == 4:
        use_lake = False
    elif observation_size == 4 + grid_size * grid_size:
        use_lake = True
    else:
        raise ValueError(
            f"Unsupported observation size: {observation_size}"
        )

    # === Evaluation Configuration ===
    seeds = LAKE_EVALUATION_SEEDS if use_lake else [42]

    env = CubeHuntEnv(
        grid_size=grid_size,
        render_mode=None,
        random_start=False,
        random_rabbit_start=True,
        random_lake=use_lake,
    )

    results = []

    try:
        for seed in seeds:

            # === Generate Reproducible Environment ===
            env.reset(seed=seed)
            lake_cells = set(env.lake_cells)

            # === Calculate Optimal Paths ===
            distances = shortest_path_distances(
                grid_size,
                lake_cells,
            )

            # === Evaluate Every Valid Rabbit Position ===
            for y in range(grid_size):
                for x in range(grid_size):

                    target = (x, y)

                    if target == (0, 0) or target in lake_cells:
                        continue

                    # Recreate the same lake for this target
                    obs, _ = env.reset(seed=seed)

                    # Position the rabbit manually
                    env.rabbit_pos = np.array(
                        target,
                        dtype=np.int32,
                    )

                    obs = env._get_obs()

                    assert env.observation_space.contains(obs)

                    blocked_moves = 0
                    success = False
                    steps = 0

                    # === Run Deterministic Policy ===
                    for step in range(
                        1,
                        MAX_EVALUATION_STEPS + 1,
                    ):
                        steps = step
                        previous_pos = env.fox_pos.copy()

                        action, _ = model.predict(
                            obs,
                            deterministic=True,
                        )

                        obs, reward, terminated, truncated, _ = (
                            env.step(int(action))
                        )

                        if np.array_equal(
                            previous_pos,
                            env.fox_pos,
                        ):
                            blocked_moves += 1

                        if terminated:
                            success = True

                        if terminated or truncated:
                            break

                    # === Evaluation Metrics ===
                    optimal_steps = distances[target]

                    results.append({
                        "lake_seed": seed if use_lake else "",
                        "lake_cells": len(lake_cells),
                        "rabbit_x": x,
                        "rabbit_y": y,
                        "success": success,
                        "steps": steps,
                        "optimal_steps": optimal_steps,
                        "extra_steps": (
                            steps - optimal_steps if success else ""
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
    print(f"Evaluated scenarios: {total}")
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

    report_path = generate_report()
    print(f"📄 Report saved: {report_path}")


if __name__ == "__main__":
    main()