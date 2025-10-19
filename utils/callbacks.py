"""
callbacks.py
-------------
Custom callbacks for Nina Project.

Author: Lucio Nunes (Nina Project)
Date: 2025-10-19
"""

from stable_baselines3.common.callbacks import BaseCallback


class EarlyStopCallback(BaseCallback):
    """
    Stops training early if the mean reward does not improve for a given number of evaluations.
    """

    def __init__(self, patience=5, min_delta=0.01, verbose=1):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -float("inf")
        self.counter = 0
        self.history = []  # incremental addition: track mean rewards per step

    def _on_step(self) -> bool:
        """
        This method is called at each step. 
        It checks if the mean reward has improved; if not, increments the counter.
        """
        # Access training metrics
        logs = self.logger.name_to_value

        if "rollout/ep_rew_mean" in logs:
            mean_reward = logs["rollout/ep_rew_mean"]
            self.history.append(mean_reward)  # record progress over time

            # Check for improvement
            if mean_reward > self.best_mean_reward + self.min_delta:
                self.best_mean_reward = mean_reward
                self.counter = 0
            else:
                self.counter += 1

            # Stop if no improvement for 'patience' evaluations
            if self.counter >= self.patience:
                if self.verbose > 0:
                    print(f"⏹️ Early stopping: no improvement for {self.patience} steps.")
                return False

        return True

    def get_training_history(self):
        """
        Returns the list of mean rewards recorded during training.
        (Added for debugging or plotting learning progress.)
        """
        return self.history
