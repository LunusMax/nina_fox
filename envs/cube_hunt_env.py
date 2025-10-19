import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

class CubeHuntEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, grid_size=10, render_mode="human"):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode

        # === Define ação e observação ===
        # Ações: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)

        # Observação: [x_fox, y_fox, x_rabbit, y_rabbit]
        low = np.array([0, 0, 0, 0], dtype=np.int32)
        high = np.array([grid_size-1, grid_size-1, grid_size-1, grid_size-1], dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # Inicializa pygame
        pygame.init()
        self.window_size = 400
        self.cell_size = self.window_size // self.grid_size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("NINA - Cubo Caçando Cubo")
        self.clock = pygame.time.Clock()

        # Estado inicial
        self.fox_pos = np.array([0, 0])
        self.rabbit_pos = np.array([self.grid_size-1, self.grid_size-1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.fox_pos = np.array([0, 0])
        self.rabbit_pos = np.array([self.grid_size-1, self.grid_size-1])
        obs = np.concatenate([self.fox_pos, self.rabbit_pos])
        return obs, {}

    def step(self, action):
        # === Mover NINA ===
        if action == 0:  # up
            self.fox_pos[1] = max(0, self.fox_pos[1] - 1)
        elif action == 1:  # right
            self.fox_pos[0] = min(self.grid_size - 1, self.fox_pos[0] + 1)
        elif action == 2:  # down
            self.fox_pos[1] = min(self.grid_size - 1, self.fox_pos[1] + 1)
        elif action == 3:  # left
            self.fox_pos[0] = max(0, self.fox_pos[0] - 1)

        # === Recompensa ===
        caught = np.array_equal(self.fox_pos, self.rabbit_pos)
        reward = 1.0 if caught else -0.01  # penaliza passo, recompensa se pegou

        obs = np.concatenate([self.fox_pos, self.rabbit_pos])
        done = bool(caught)  # termina episódio se pegou o coelho

        return obs, reward, done, False, {}

    def render(self):
        self.screen.fill((50, 120, 50))
        # grid
        for x in range(0, self.window_size + 1, self.cell_size):
            pygame.draw.line(self.screen, (70, 70, 70), (x, 0), (x, self.window_size), 1)
        for y in range(0, self.window_size + 1, self.cell_size):
            pygame.draw.line(self.screen, (70, 70, 70), (0, y), (self.window_size, y), 1)
        # coelho
        pygame.draw.rect(self.screen, (0, 0, 0),
                         (self.rabbit_pos[0]*self.cell_size,
                          self.rabbit_pos[1]*self.cell_size,
                          self.cell_size, self.cell_size))
        # NINA
        pygame.draw.rect(self.screen, (255, 140, 0),
                         (self.fox_pos[0]*self.cell_size,
                          self.fox_pos[1]*self.cell_size,
                          self.cell_size, self.cell_size))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
