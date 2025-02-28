"""Provides logic that allows to wrap environments with additional functionality."""

from typing import Any, SupportsFloat
import gymnasium as gym


class ActionStatsWrapper(gym.Wrapper):
    """A wrapper that logs the actions taken by the agent."""

    def step(self, action) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        *transition, info = super().step(action)

        # report each dim of the action
        action_stats = {f"action_{i}": a for i, a in enumerate(action)}

        info.update(action_stats)

        return (*transition, info)  # type: ignore
