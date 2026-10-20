
import pygame

from envs.cube_hunt_env import CubeHuntEnv


def main():
    """Preview the environment using random actions."""

    env = CubeHuntEnv(render_mode="human", random_start=True, random_rabbit_start=True, random_lake=True,)
    obs, info = env.reset()

    try:
        running = True

        while running:
            # Handle window events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if not running:
                break

            # Choose a random action
            action = env.action_space.sample()

            # Update environment state
            obs, reward, terminated, truncated, info = env.step(action)

            # Render the updated state
            env.render()

            # Start another episode when finished
            if terminated or truncated:
                obs, info = env.reset()

    finally:
        env.close()


if __name__ == "__main__":
    main()
