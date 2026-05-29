"""Per-episode summary wrapper for race_car training.

On the terminating step of each episode, attaches a summary dict to
``info["train_episode"]`` containing one row per finished lap:

    {
        "lap_time":   steps * dt,
        "lap_return": cumulative reward over the episode,
        "steps":      number of steps in the episode,
        "success":    bool, lap completed,
        "violation":  bool, off-track truncation,
    }

The SAC trainer routes this dict through ``report_stats("train_episode", ...)``
without smoothing, so each finished lap produces one row in
``train_episode_log.csv`` (and one tensorboard / wandb point). This is the
truthful per-episode signal the moving-window step stats can't show.
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym

from leap_c.examples.race_car.env import RaceCarEnv


class RaceCarEpisodeStats(gym.Wrapper):
    """Emit one ``info['train_episode']`` summary on each episode boundary."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        race_env = env.unwrapped
        if not isinstance(race_env, RaceCarEnv):
            raise TypeError(
                f"RaceCarEpisodeStats expects RaceCarEnv, got {type(race_env).__name__}"
            )
        self._dt = float(race_env.cfg.dt)
        self._steps = 0
        self._return = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        self._steps = 0
        self._return = 0.0
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        self._return += float(reward)

        if terminated or truncated:
            task = info.get("task", {})
            info["train_episode"] = {
                "lap_time": self._steps * self._dt,
                "lap_return": self._return,
                "steps": self._steps,
                "success": float(bool(task.get("success", False))),
                "violation": float(bool(task.get("violation", False))),
            }
        return obs, reward, terminated, truncated, info
