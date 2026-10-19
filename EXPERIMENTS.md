# Nina Fox — Experiment Log

This document records **what changed, why it changed, and what the experiment actually showed**. Implemented changes, observed results, and future plans are kept separate. A demonstration video is not, by itself, a reliable performance metric.

## Experiment index

| Version | Main change | Requested training timesteps | Status | Observed outcome |
| --- | --- | ---: | --- | --- |
| `v0.1` | Fixed starting positions | 20,000 | Completed | Nina reached the fixed rabbit position in the observed demonstration. |
| `v0.2` | Random rabbit starting position; Nina fixed at `(0, 0)` | 20,000 | Completed; unsuccessful evaluation | Nina repeatedly chose `LEFT` at `(0, 0)` and did not capture the rabbit. |
| `v0.2.1` | Manhattan-distance reward shaping | **To verify** | Executed; unsuccessful evaluation | Nina moved right across the top row, became stuck at `(9, 0)`, and did not capture the rabbit within 500 steps. |
| `v0.3` | Limit episodes; evaluate systematically; improve training protocol | TBD | **Planned** | Not tested. |

---

## Machines and training time

### Machine A

| Specification | Value |
| --- | --- |
| Identifier | `A` |
| CPU | TBD |
| GPU | TBD |
| RAM | TBD |
| Operating system | TBD |
| Python | TBD |
| PyTorch | TBD |
| Stable-Baselines3 | TBD |
| Gymnasium | TBD |
| Notes | TBD |

### Machine B

| Specification | Value |
| --- | --- |
| Identifier | `B` |
| CPU | AMD Ryzen 7 8700G (8 cores / 16 threads) |
| GPU | NVIDIA GeForce RTX 5060 (8 GB VRAM) |
| Integrated GPU | AMD Radeon 780M |
| RAM | 64 GB (62 GiB usable) |
| Operating system | Ubuntu 24.04.4 LTS |
| Kernel | Linux 7.0.0-30-generic |
| Python | 3.12.3 |
| PyTorch | 2.14.0+cu130 |
| Stable-Baselines3 | 2.9.0 |
| Gymnasium | 1.3.0 |
| NVIDIA Driver | 595.99.02 |
| CUDA available | Yes |
| Notes | PPO training device must be recorded per experiment. |

### Training-duration history

| Version | Machine | Training time | Device used by PPO | Measurement notes |
| --- | --- | --- | --- | --- |
| `v0.1` | `B` | Not recorded | Not recorded | Historical run; duration was not measured. |
| `v0.2` | `B` | Not recorded | Not recorded | Historical run; duration was not measured. |
| `v0.2.1` | `B` | Not recorded | Not recorded | Historical run; duration was not measured. |
| `v0.3` | `B` | 00:33:31 | CUDA (RTX 5060) | 100,096 timesteps; includes training timelapse generation (10,009 frames). |

**Timing protocol:** measure the wall-clock duration of `model.learn()` using `time.perf_counter()`. This includes training callbacks, rendering/capture overhead, and saving the timelapse in `_on_training_end()`. It **does not** include the later test, model save, or evaluation video. Record the exact code commit, requested timesteps, effective timesteps (`model.num_timesteps`), chosen machine (`A`/`B`), PPO device, and random seed for each run. Durations measured on different machines should **not** be used to claim algorithmic speed improvements without a controlled benchmark.

**Hardware collection (run separately on machines A and B):**

```bash
lscpu | grep 'Model name'
free -h
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 --version
python3 - <<'PY'
import torch, stable_baselines3, gymnasium
print('PyTorch:', torch.__version__)
print('Stable-Baselines3:', stable_baselines3.__version__)
print('Gymnasium:', gymnasium.__version__)
PY
```

---

## v0.1 — Fixed-target baseline

**Status:** Completed.

**Goal:** Establish that PPO can learn navigation in the simplest environment.

### Environment

- Grid: `10 × 10`.
- Nina starts at `(0, 0)`.
- Rabbit starts at `(9, 9)` and remains stationary throughout the episode.
- Actions: `UP`, `RIGHT`, `DOWN`, `LEFT`.
- Observation: `[fox_x, fox_y, rabbit_x, rabbit_y]`.
- Reward: `+1.0` for capture; `-0.01` for other steps.

### Training

- Algorithm / policy: PPO / `MlpPolicy`.
- Requested timesteps: `20,000`.
- Learning rate: `3e-4`.
- PPO rollout length (`n_steps`): `256`.
- Batch size: `64`.
- Discount factor (`gamma`): `0.99`.
- Machine and duration: **not recorded**.

### Observation

Nina was observed reaching the rabbit at its fixed position. This does **not** demonstrate generalization: a fixed sequence of actions could solve this version.

### Artifacts

- Model: `nina_logs/models/nina_ppo_model_v0.1` (historical filename; may later be renamed with `.zip`).
- Evaluation video: see the stored artifact for this version (historical filenames may differ).

---

## v0.2 — Random initial rabbit position

**Status:** Implemented and evaluated; observed failure.

**Goal:** Test whether Nina learns a policy conditioned on the rabbit's coordinates.

### Changes

- At each `reset()`, keep Nina at `(0, 0)`.
- Randomize the rabbit's initial position, excluding Nina's cell.
- Keep the rabbit stationary during an episode.
- Preserve sparse reward: `+1.0` on capture and `-0.01` otherwise.
- Requested training timesteps: `20,000`.
- Machine and duration: **not recorded**.

**Relevant files:** `envs/cube_hunt_env.py`, `nina_brain/agents/train_ppo.py`, `utils/video_recorder.py`.

### Observations

- The timelapse contained `2,022` recorded frames.
- Post-training test: failed to catch the rabbit within `500` steps.
- Deterministic 10-step test: `LEFT` on all 10 steps, remaining at `(0, 0)`.
- The evaluated policy stalled at the left boundary, where repeated invalid moves left observations unchanged.

### Interpretation

Sparse rewards may contribute to failure: approaching, retreating, and pressing against a boundary all result in `-0.01` until capture. This is a hypothesis, **not an isolated causal result**.

---

## v0.2.1 — Manhattan-distance reward shaping

**Status:** Executed; unsuccessful single-episode evaluation.

**Goal:** Provide feedback for moving toward the rabbit rather than solely rewarding capture.

### Reward change

The Manhattan distance is

\[
D(s)=|x_{fox}-x_{rabbit}|+|y_{fox}-y_{rabbit}|.
\]

For nonterminal steps:

\[
r=-0.01+0.02\left(D_{before}-D_{after}\right).
\]

Capture remains worth `+1.0`.

| Movement outcome | Immediate reward |
| --- | ---: |
| One cell closer (without capture) | `+0.01` |
| Blocked by a wall | `-0.01` |
| One cell farther away | `-0.03` |
| Capture | `+1.00` |

**Research note:** With `gamma=0.99`, this is not exactly discount-consistent potential-based shaping; preserving the original optimal policy is not guaranteed. Record it as a change to reward design.

### Training configuration

- Algorithm / policy: PPO / `MlpPolicy`.
- Requested timesteps: **confirm from the script or training log used for this run**.
- Machine and duration: **not recorded**.
- Effective training timesteps: **not explicitly logged**.
- **Consistency check:** with `TRAIN_TIMELAPSE_INTERVAL=10`, `2,022` captured frames correspond to approximately `20,220` training steps. PPO's `n_steps=256` and a request for 20,000 steps would normally produce `20,224` effective steps. This is **consistent with a 20,000-step run**, not a 100,000-step run; confirm that the interval was unchanged in the executed configuration.

### Observed results

Terminal output:

```text
🎞️ Saving training timelapse (2022 frames)...
✅ Training timelapse saved as nina_logs/videos/nina_training_timelapse_v0.2.1.mp4
⚠️ Nina did not catch the rabbit within the step limit.
💾 Model saved in nina_logs/models/nina_ppo_model_v0.2.1.zip
🎬 Starting video recording (nina_logs/videos/nina_learning_v0.2.1.mp4) ...
⏱️ Recording ended before capture.
🎥 Recording 500 frames...
✅ Video saved as nina_logs/videos/nina_learning_v0.2.1.mp4
✅ Training complete.
```

Video inspection: Nina moved from `(0, 0)` to `(9, 0)` along the top edge, then remained stuck at the right boundary. The rabbit appeared near `(9, 7)` and was not caught during the 500-frame recording. This differs from v0.2's `LEFT`-at-origin failure, but does **not** establish robust learning.

### Artifacts

- Model: `nina_logs/models/nina_ppo_model_v0.2.1.zip`.
- Evaluation episode: `nina_logs/videos/nina_learning_v0.2.1.mp4`.
- Training timelapse: `nina_logs/videos/nina_training_timelapse_v0.2.1.mp4`.
- Mean reward, capture rate over multiple episodes, random seeds, and training duration: **not recorded**.

### Conclusion

Reward shaping alone, at the settings actually executed, did not prevent boundary-stalling in the deterministic evaluation. Do not claim this version succeeded without a multi-episode evaluation.

---

## v0.3 — Proposed: finite episodes and systematic evaluation

**Status:** Planned; not implemented or tested.

### Problem

The current environment can continue an episode indefinitely when no capture occurs. A deterministic policy can repeatedly choose a blocked action at a boundary and receive an unchanged observation forever. A 500-step cap in the post-training test or video recorder does **not** itself constrain the episodes used during PPO training.

### Proposed changes

1. **Limit training episodes** to a documented maximum (initial proposal: `100` steps), e.g. with `gymnasium.wrappers.TimeLimit`. Reaching the limit must set `truncated=True`; catching the rabbit must set `terminated=True`.
2. **Keep random rabbit positions** during training and evaluation; keep the rabbit stationary during each episode. Verify that `video_recorder.py` gets the same environment settings.
3. **Record exact training duration and machine** using the protocol above, plus the effective steps, algorithm device, and seed.
4. **Evaluate at least 100 seeded episodes** with `deterministic=True`, reporting capture rate, mean successful capture length, mean return, and boundary-stall incidence.
5. **If the baseline still fails**, test increased timesteps (e.g., `100,000`) and/or a modified reward policy as *separate documented experiments* so individual effects can be assessed.

### Results — fill after executing

| Field | Result |
| --- | --- |
| Date and Git commit | TBD |
| Machine | `A` / `B` (TBD) |
| Training time (`model.learn()`) | TBD |
| Training device | TBD |
| Training seed(s) | TBD |
| Requested / effective timesteps | TBD |
| Maximum episode length | TBD |
| Evaluation episodes / seed(s) | TBD |
| Capture rate | TBD |
| Mean successful capture steps | TBD |
| Mean episode return | TBD |
| Boundary-stall failures | TBD |
| Model / episode video / timelapse | TBD |
| Conclusion | TBD |

---

## Naming and reproducibility conventions

- `VERSION` names a configuration/experiment, **not** a counter for re-running a script.
- Use `nina_logs/models/nina_ppo_model_vX.Y.zip` for saved PPO models.
- Use `nina_logs/videos/nina_learning_vX.Y.mp4` for evaluation video.
- Use `nina_logs/videos/nina_training_timelapse_vX.Y.mp4` for training timelapse.
- Re-running the same version with the same output paths can overwrite existing artifacts. Use a separate run ID or seed suffix to preserve distinct repetitions.
- Build a video filename by appending `.mp4` to the whole name, not `Path.with_suffix('.mp4')` on a name ending in `v0.2`.
- Preserve the source commit, environment settings, random seed, PPO hyperparameters, Python/library versions, machine identifier, device, exact effective timesteps, duration, and evaluation protocol.
- A single video is a demonstration. A multi-episode capture rate is a performance metric.

---

## Template — copy for each future experiment

```markdown
## vX.Y — Experiment title

**Status:** Planned / Implemented / Evaluated
**Date:** YYYY-MM-DD
**Git commit:** `<commit>`

### Problem and hypothesis
- Problem: ...
- Hypothesis: ...

### Change
- Files: ...
- Before / after: ...
- Unchanged controls: ...

### Training setup
| Field | Value |
| --- | --- |
| Machine | A / B |
| CPU / GPU / RAM | See machine registry |
| PPO device | CPU / CUDA |
| Python / Torch / SB3 | ... |
| Seed | ... |
| Requested / effective timesteps | ... |
| Hyperparameters | ... |
| Environment and episode limit | ... |
| Training time (includes callbacks) | ... |

### Evaluation
- Number of seeded episodes: ...
- Capture rate: ...
- Mean steps on successful episodes: ...
- Mean return: ...
- Boundary-stall cases: ...

### Artifacts
- Model: ...
- Evaluation video: ...
- Training timelapse: ...

### Conclusion and next question
...
```
