"""
Nina Fox — experiment report generator.

Usage (from the project root):
    python3 -m utils.report_generator
    python3 -m utils.report_generator --version v0.3

The default version is read from nina_brain/agents/train_ppo.py.
Evaluation metrics come from nina_logs/evaluations/evaluation_<version>.csv.
Training metadata, when recorded, comes from
nina_logs/metadata/nina_training_<version>.json.

The Conclusion section is preserved when a report is regenerated.
"""

import argparse
import ast
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "nina_brain" / "agents" / "train_ppo.py"
REPORTS = ROOT / "nina_logs" / "reports"

CONCLUSION_START = "<!-- NINA_CONCLUSION_START -->"
CONCLUSION_END = "<!-- NINA_CONCLUSION_END -->"


def _script_constants():
    """Read simple top-level constants without importing the training script."""
    tree = ast.parse(TRAIN_SCRIPT.read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if names:
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError):
                    continue
                for name in names:
                    values[name] = value
    return values


def current_version():
    version = _script_constants().get("VERSION")
    if not isinstance(version, str) or not re.fullmatch(r"v[0-9]+(?:\.[0-9]+)*", version):
        raise ValueError("Could not read a valid VERSION from train_ppo.py")
    return version


def _paths(version):
    return {
        "csv": ROOT / "nina_logs" / "evaluations" / f"evaluation_{version}.csv",
        "model": ROOT / "nina_logs" / "models" / f"nina_ppo_model_{version}.zip",
        "video": ROOT / "nina_logs" / "videos" / f"nina_learning_{version}.mp4",
        "timelapse": ROOT / "nina_logs" / "videos" / f"nina_training_timelapse_{version}.mp4",
        "metadata": ROOT / "nina_logs" / "metadata" / f"nina_training_{version}.json",
        "report": REPORTS / f"nina_report_{version}.md",
    }


def _read_metadata(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_training_metadata(model, duration_seconds, *, version=None, machine=None,
                           post_training_steps=None):
    """Call after training to persist facts that the evaluation CSV cannot know."""
    version = version or current_version()
    path = _paths(version)["metadata"]
    path.parent.mkdir(parents=True, exist_ok=True)

    values = _script_constants() if version == current_version() else {}
    device = str(model.device)

    if device.startswith("cuda"):
        try:
            import torch
            device = f"CUDA ({torch.cuda.get_device_name(model.device)})"
        except (ImportError, RuntimeError, ValueError, AssertionError):
            device = "CUDA"
    elif device.startswith("cpu"):
        device = "CPU"

    metadata = {
        "machine": machine or os.getenv("NINA_MACHINE") or platform.node(),
        "algorithm": "PPO (MlpPolicy)",
        "requested_timesteps": values.get("TOTAL_TIMESTEPS"),
        "effective_timesteps": int(model.num_timesteps),
        "duration_seconds": float(duration_seconds),
        "training_device": device,
    }
    if post_training_steps is not None:
        metadata["post_training_steps"] = int(post_training_steps)

    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _model_timesteps(model_path):
    """SB3 models usually contain `num_timesteps` in ZIP member `data`."""
    if not model_path.exists():
        return None
    try:
        with zipfile.ZipFile(model_path) as archive:
            data = json.loads(archive.read("data"))
        return int(data["num_timesteps"])
    except (OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile):
        return None


def _video_frame_count(path):
    """Read MP4 frame count if ffprobe is installed and the file exists."""
    if not path.exists() or shutil.which("ffprobe") is None:
        return None
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(path),
    ]
    try:
        result = subprocess.run(cmd, text=True, capture_output=True,
                                timeout=10, check=True)
        return int(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        return None


def _read_evaluation(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation CSV not found: {path}\n"
            "Run: python3 -m nina_brain.evaluate"
        )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"success", "steps", "extra_steps", "blocked_moves"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing evaluation columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("Evaluation CSV contains no scenarios.")

    successes = [
        row for row in rows
        if str(row["success"]).strip().lower() in {"true", "1", "yes"}
    ]
    n_success = len(successes)
    extras = [float(row["extra_steps"]) for row in successes]
    steps = [float(row["steps"]) for row in successes]

    return {
        "total": len(rows),
        "successes": n_success,
        "failures": len(rows) - n_success,
        "rate": 100 * n_success / len(rows),
        "mean_steps": mean(steps) if steps else None,
        "mean_extra": mean(extras) if extras else None,
        "optimal": sum(abs(n) < 1e-9 for n in extras),
        "blocked": sum(int(row["blocked_moves"]) for row in rows),
        "lakes": len({
            row["lake_seed"] for row in rows
            if row.get("lake_seed", "").strip()
        }),
    }


def _duration(seconds):
    if seconds is None:
        return "Not recorded"
    seconds = round(float(seconds))
    return f"{seconds // 3600:02}:{(seconds // 60) % 60:02}:{seconds % 60:02}"


def _value(value):
    return "Not recorded" if value is None or value == "" else str(value)


def _existing_conclusion(path):
    if not path.exists():
        return "\n**Conclusion:** TODO — interpret the results and state the next hypothesis.\n"
    previous = path.read_text(encoding="utf-8")
    match = re.search(
        re.escape(CONCLUSION_START) + r"(.*?)" + re.escape(CONCLUSION_END),
        previous, flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(
            f"Existing report has no conclusion markers: {path}. "
            "File not overwritten to protect manual edits."
        )
    return match.group(1)


def generate_report(version=None):
    version = version or current_version()
    if not re.fullmatch(r"v[0-9]+(?:\.[0-9]+)*", version):
        raise ValueError(f"Invalid version: {version}")

    paths = _paths(version)
    evaluation = _read_evaluation(paths["csv"])
    metadata = _read_metadata(paths["metadata"])
    REPORTS.mkdir(parents=True, exist_ok=True)

    effective = metadata.get("effective_timesteps")
    if effective is None:
        effective = _model_timesteps(paths["model"])

    requested = metadata.get("requested_timesteps")
    if requested is None and version == current_version():
        requested = _script_constants().get("TOTAL_TIMESTEPS")

    frames = _video_frame_count(paths["timelapse"])
    recorded_steps = _video_frame_count(paths["video"])

    optimal_percent = (
        100 * evaluation["optimal"] / evaluation["successes"]
        if evaluation["successes"] else 0.0
    )

    rows = [
        ("Machine", _value(metadata.get("machine"))),
        ("Algorithm", _value(metadata.get("algorithm", "PPO (MlpPolicy)"))),
        ("Requested training timesteps", _value(requested)),
        ("Effective training timesteps", _value(effective)),
        ("Training duration", _duration(metadata.get("duration_seconds"))),
        ("Training device", _value(metadata.get("training_device"))),
        ("Training timelapse frames", _value(frames)),
        ("Post-training evaluation steps", _value(metadata.get("post_training_steps"))),
        ("Recorded episode frames", _value(recorded_steps)),
        ("Evaluated scenarios", evaluation["total"]),
        ("Successful captures", evaluation["successes"]),
        ("Failed captures", evaluation["failures"]),
        ("Success rate", f'{evaluation["rate"]:.2f}%'),
        ("Mean capture steps (successes)", _value(
            f'{evaluation["mean_steps"]:.2f}' if evaluation["mean_steps"] is not None else None
        )),
        ("Mean extra steps (successes)", _value(
            f'{evaluation["mean_extra"]:.2f}' if evaluation["mean_extra"] is not None else None
        )),
        ("Optimal-path captures", evaluation["optimal"]),
        ("Optimal captures among successes", f"{optimal_percent:.2f}%"),
        ("Total blocked movements", evaluation["blocked"]),
        ("Distinct lake configurations", evaluation["lakes"]),
        ("Model", f'`{paths["model"].name}`'),
        ("Training video", f'`{paths["timelapse"].name}`'),
        ("Evaluation video", f'`{paths["video"].name}`'),
        ("Evaluation dataset", f'`{paths["csv"].name}`'),
    ]

    table = "\n".join(
        ["| **Metric** | **Value** |", "| --- | --- |"]
        + [f"| {name} | {value} |" for name, value in rows]
    )

    conclusion = _existing_conclusion(paths["report"])
    markdown = (
        f"### {version} — Training and Evaluation Results\n\n"
        + table
        + "\n\n"
        + CONCLUSION_START
        + conclusion
        + CONCLUSION_END
        + "\n"
    )

    paths["report"].write_text(markdown, encoding="utf-8")
    return paths["report"]


def main():
    parser = argparse.ArgumentParser(description="Generate Nina Fox experiment report.")
    parser.add_argument("--version", help="Version such as v0.4 (default: train_ppo.py)")
    args = parser.parse_args()
    path = generate_report(args.version)
    print(f"✅ Report generated: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
