"""Contains the necessary functions for validation."""

from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Any, Generator

import torch
from gymnasium import Env
from gymnasium.wrappers import RecordVideo
from timeit import default_timer


def episode_rollout(
    policy: Callable,
    env: Env,
    episodes: int = 1,
    render_episodes: int = 0,
    render_human: bool = False,
    video_folder: Optional[str | Path] = None,
    name_prefix: Optional[str] = None,
) -> Generator[tuple[dict[str, bool | Any], defaultdict[Any, list]], Any, None]:
    """Rollout an episode and returns the cumulative reward.

    Args:
        policy (Callable): The policy to be used for the rollout.
        env (Env): The gym environment.
        episodes (int): The number of episodes to run.
        render_episodes (int): The number of episodes to render. If 0, no episodes
        render_human (bool): If True, render the environment should be in human render
            mode. Can not be true if video_path is set.
        video_folder (Optional[str | Path]): The environment is rendered and saved as a
            video in this folder. Can not be set if render_human is True.
        name_prefix (Optional[str]): The prefix for the video file names. Must be set if
            video_folder is set.

    Returns:
        A dictionary containing the information about the rollout, at containing the
        keys "score", "length", "terminated", and "truncated" and a dictionary of
        policy statistics.
    """
    if render_human and video_folder is not None:
        raise ValueError("render_human and video_path can not be set at the same time.")
    if video_folder is not None and name_prefix is None:
        raise ValueError("name_prefix must be set if video_path is set.")

    def render_trigger(episode_id):
        return episode_id < render_episodes

    if video_folder is not None:
        env = RecordVideo(
            env, video_folder, name_prefix=name_prefix, episode_trigger=render_trigger
        )

    with torch.no_grad():
        for episode in range(episodes):
            policy_stats = defaultdict(list)
            episode_stats = defaultdict(list)
            o, _ = env.reset()

            terminated = False
            truncated = False

            cum_inference_time = 0.0

            while not terminated and not truncated:
                t0 = default_timer()
                a, stats = policy(o)
                cum_inference_time += default_timer() - t0

                if stats is not None:
                    for key, value in stats.items():
                        policy_stats[key].append(value)

                if isinstance(a, torch.Tensor):
                    a = a.cpu().numpy()

                o_prime, r, terminated, truncated, info = env.step(a)

                if "task" in info:
                    for key, value in info["task"].items():
                        episode_stats[key].append(value)

                if render_human and render_trigger(episode):
                    env.render()

                o = o_prime

            assert "episode" in info, (
                "The environment did not return episode information."
            )
            rollout_stats = {
                "score": info["episode"]["r"],
                "length": info["episode"]["l"],
                "terminated": terminated,
                "truncated": truncated,
                "inference_time": cum_inference_time / info["episode"]["l"],
            }
            rollout_stats.update(episode_stats)

            yield rollout_stats, policy_stats

        env.close()
