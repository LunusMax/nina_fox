# 🧠 Nina Project

Nina Project is a reinforcement learning sandbox built in **Python** using **Gymnasium** and **Stable-Baselines3**.  
The current stage features a simple 2D environment where *Nina* (an orange cube) learns to chase a black cube inside a grid world.

---

## 🚀 Features

- Custom **Gymnasium environment** (`CubeHuntEnv`)
- **PPO** agent training with Stable-Baselines3
- **Video recorder** for evaluation episodes
- **Training timelapse callback** (AI-Warehouse-style visualization)
- Modular project structure for easy extension to 3D environments

---

## 🧩 Project Structure

```
nina_fox/
│
├── nina_brain/
│   └── agents/
│       └── train_ppo.py             # Main training script
│
├── envs/
│   └── cube_hunt_env.py             # Custom 2D environment
│
├── utils/
│   ├── video_recorder.py            # Records test episodes
│   ├── training_recorder_callback.py# Records training timelapse
│   └── callbacks.py                 # Custom early-stop callback
│
└── nina_logs/
    ├── models/                      # Saved PPO models
    └── videos/                      # Saved MP4 timelapses
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/LunusMax/nina_fox.git
cd nina_fox

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate    # (Windows)
# source .venv/bin/activate  (Linux/Mac)

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🧠 Train Nina

Run the PPO training loop:

```bash
python nina_brain/agents/train_ppo.py
```

During training:
- Nina learns via **PPO** on the grid world.  
- A **timelapse video** of the learning process is automatically generated at `nina_logs/videos/`.  
- After training, a **final evaluation video** and the **trained model** are saved.

---

## 🛠 Requirements

- Python ≥ 3.10  
- Gymnasium ≥ 0.29  
- Stable-Baselines3 ≥ 2.3  
- Pygame ≥ 2.6  
- ImageIO with FFmpeg (`pip install imageio[ffmpeg]`)

---

## 📈 Next Steps

- Expand Nina’s environment into **3D simulation**
- Implement additional RL algorithms (DQN, SAC)
- Add reward visualization and performance tracking tools

---

## 🧑‍💻 Author

**Lucio Nunes**  
_Nina Project (2025)_
