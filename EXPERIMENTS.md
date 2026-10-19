# Nina Fox — Experiment Log

This document records changes to Nina Fox's reinforcement-learning environment, training configuration, and evaluation results. Its purpose is to explain **what changed, why it changed, and what the experiment actually showed**.

> **Documentation rule:** Separate *implemented changes*, *observed results*, and *planned experiments*. Do not mark an experiment as successful until its outcome has been measured.

## Experiment index

| Version | Main change | Training steps | Status | Evidence / outcome |
| --- | --- | ---: | --- | --- |
| `v0.1` | Fixed initial positions | 20,000 | Completed | Nina learned to reach the fixed rabbit position in the observed demonstration. |
| `v0.2` | Random rabbit starting position at each reset; Nina starts at `(0, 0)` | 20,000 | Completed; unsuccessful evaluation | After training, deterministic evaluation failed to catch the rabbit within 500 steps. At `(0, 0)`, Nina chose `LEFT` repeatedly. |
| `v0.2.1` | Reward shaping based on Manhattan-distance improvement | **To be defined** | **Proposed — not yet verified** | Hypothesis: denser reward feedback should help Nina learn to approach rabbits at varying positions. |

---

## v0.1 — Fixed-target baseline

**Status:** Completed  
**Goal:** Establish that PPO can learn navigation in the simplest environment.

### Environment

- Grid: `10 × 10`.
- Nina starts at `(0, 0)`.
- Rabbit starts at `(9, 9)` and remains stationary throughout the episode.
- Actions: `UP`, `RIGHT`, `DOWN`, `LEFT`.
- Observation: `[fox_x, fox_y, rabbit_x, rabbit_y]`.
- Reward: `+1.0` for capture; `-0.01` for other steps.

### Training

- Algorithm: PPO (`MlpPolicy`).
- Total timesteps: `20,000`.
- Learning rate: `3e-4`.
- Rollout length (`n_steps`): `256`.
- Batch size: `64`.
- Discount factor (`gamma`): `0.99`.

### Observation

Nina was observed learning to reach the rabbit at its fixed position. This baseline does **not** demonstrate generalization to other rabbit locations; a fixed sequence of moves can solve the task.

### Artifacts

- Model: `nina_logs/models/nina_ppo_model_v0.1` (the observed stored filename; may differ if later saved with `.zip`).
- Episode video: `nina_logs/videos/nina_learning_v0.1.mp4` (desired naming convention; confirm the local artifact).

### Lesson

The environment and PPO training loop can support learning a fixed-target task. The next experiment must test whether the learned policy uses the rabbit's coordinates rather than memorizing one route.

---

## v0.2 — Random initial rabbit position

**Status:** Implemented and tested; deterministic evaluation failed  
**Goal:** Test whether Nina can learn to approach a rabbit placed at different coordinates.

### Change

- At each episode reset, keep Nina at `(0, 0)`.
- Sample a new rabbit position, different from Nina's position.
- Keep the rabbit stationary during the episode.
- Preserve the sparse reward: `+1.0` for capture and `-0.01` otherwise.
- Keep PPO training at `20,000` timesteps for an initial comparison with `v0.1`.

**Relevant files:**

- `envs/cube_hunt_env.py` — rabbit initialization in `reset()`.
- `nina_brain/agents/train_ppo.py` — experiment version and environment configuration.
- `utils/video_recorder.py` — match recording conditions to training conditions.

### Observed results

- Training generated a timelapse containing **2,022 captured frames**.
- The post-training evaluation reported: `Nina did not catch the rabbit within the step limit` (`500` steps).
- The recorded evaluation showed Nina remaining at `(0, 0)`.
- A separate deterministic 10-action test at `(0, 0)` produced `LEFT` on **all 10 steps**, leaving Nina at `(0, 0)` throughout.
- During the video inspection, the rabbit was at `(9, 9)`. **Check that the evaluation recorder actually received the randomized-rabbit setting**; the fixed target in the video alone cannot establish a recorder bug.

### Interpretation

The evaluation policy became stuck at the left boundary. Because invalid moves do not change the observation, a deterministic policy that keeps selecting `LEFT` receives the same input each step and repeats the action.

A likely contributor is **sparse reward feedback**: moving closer, moving farther, and hitting a wall all produce `-0.01` until capture. This is a hypothesis about *why learning failed*, not yet an experimentally isolated cause.

### Next action

Change the reward signal in a separate experiment (`v0.2.1`), retain a reproducible evaluation protocol, and measure the capture rate across many randomized episodes.

---

## v0.2.1 — Manhattan-distance reward shaping

**Status:** **PROPOSED — do not label as implemented until code is changed and executed**  
**Goal:** Give the agent immediate feedback for approaching the rabbit.

### Hypothesis

Adding a small reward for reducing the Manhattan distance to the rabbit may reduce the tendency to learn ineffective actions, including repeatedly moving into a boundary.

### Proposed modification

File: `envs/cube_hunt_env.py`, method: `step()`.

Define the Manhattan distance:

\[
D(s) = |x_{fox} - x_{rabbit}| + |y_{fox} - y_{rabbit}|
\]

For a nonterminal step, calculate:

\[
r = -0.01 + 0.02\left(D_{before} - D_{after}\right)
\]

Keep the capture reward at `+1.0`.

```python
old_distance = np.abs(self.fox_pos - self.rabbit_pos).sum()

# Apply Nina's selected movement here.

new_distance = np.abs(self.fox_pos - self.rabbit_pos).sum()
caught = np.array_equal(self.fox_pos, self.rabbit_pos)

if caught:
    reward = CATCH_REWARD
else:
    distance_improvement = old_distance - new_distance
    reward = STEP_PENALTY + 0.02 * distance_improvement
```

Expected immediate rewards:

| Action outcome | Reward |
| --- | ---: |
| One cell closer to rabbit (without capture) | `+0.01` |
| No change in distance (e.g., blocked by wall) | `-0.01` |
| One cell farther from rabbit | `-0.03` |
| Capture | `+1.00` |

> **Research note:** This shaping rule changes the reward function. With `gamma = 0.99`, it is not exactly the same as discount-consistent potential-based shaping, so preserving the original task's optimal policy is **not guaranteed**. We should report it explicitly as a reward-design change.

### Experiment controls

To isolate the effect of reward shaping, **first keep `TOTAL_TIMESTEPS = 20_000`**, with the same grid, PPO hyperparameters, and environment initialization as `v0.2`. If learning is still insufficient, test `100_000` steps as a **separate** experiment (for example, `v0.2.2`) so training duration does not confound the comparison.

Proposed training configuration:

```python
VERSION = "v0.2.1"
TOTAL_TIMESTEPS = 20_000
```

### Evaluation protocol (to run)

Evaluate the saved model over **100 episodes** with distinct seeded rabbit positions and `deterministic=True`. Report:

- **Capture rate:** successful episodes / 100.
- **Mean steps to capture:** calculated for successful episodes only, with the denominator stated.
- **Mean episode return:** average sum of rewards per episode.
- **Boundary-stall count:** number of episodes with repeated blocked moves.
- **Seed(s):** to make outcomes reproducible.

Use a finite time limit in the environment (or explicitly cap evaluation steps) so failure episodes are recorded consistently. A training-time limit, if added, should return `truncated=True`, not a successful termination.

### Results — fill after running

- Date: `TBD`
- Git commit: `TBD`
- Training seed: `TBD`
- Timesteps actually trained: `TBD`
- Capture rate (100 episodes): `TBD`
- Mean steps to capture: `TBD`
- Mean episode return: `TBD`
- Observed failure modes: `TBD`
- Model: `TBD`
- Episode video: `TBD`
- Training timelapse: `TBD`
- Conclusion: `TBD`

---

## Naming and reproducibility conventions

- `VERSION` names the experiment, **not** the number of times a script has run.
- Prefer one versioned model and one video of each type per experiment:
  - `nina_logs/models/nina_ppo_model_vX.Y.zip`
  - `nina_logs/videos/nina_learning_vX.Y.mp4`
  - `nina_logs/videos/nina_training_timelapse_vX.Y.mp4`
- Re-running a version with identical output paths **overwrites its artifacts**. Change `VERSION` before intentionally preserving a new experiment.
- For filenames where `v0.2` is part of the stem, **do not** use `Path.with_suffix('.mp4')` on `nina_learning_v0.2`: it interprets `.2` as a suffix. Append `.mp4` directly instead.
- Record source commit, environment configuration, seed, total training steps, hyperparameters, and evaluation results.
- A single successful video is a demonstration; **capture rate across multiple episodes** is the performance measure.
- Keep evaluation conditions aligned with training conditions unless testing out-of-distribution generalization deliberately.

---

## Template — copy for every future change

```markdown
## vX.Y — Short experiment title

**Status:** Proposed / Implemented / Evaluated  
**Date:** YYYY-MM-DD  
**Git commit:** `<hash>`

### Problem

What failed or what question remains unanswered?

### Hypothesis

What specific improvement do we expect, and why?

### Changes

- **Files:** `path/to/file.py`
- **Before:** ...
- **After:** ...
- **Unchanged controls:** ...

### Training configuration

- Algorithm / policy: ...
- Seed(s): ...
- Timesteps: ...
- Hyperparameters: ...
- Environment settings: ...

### Evaluation

- Number of episodes: ...
- Capture rate: ...
- Mean steps to capture: ...
- Mean episode return: ...
- Failure cases: ...

### Artifacts

- Model: ...
- Evaluation video: ...
- Training timelapse: ...

### Conclusion / Next question

What did the evidence show? What should be changed or tested next?
```
