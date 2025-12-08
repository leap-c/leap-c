"""Contains the necessary functions for validation."""

from collections import defaultdict
from pathlib import Path
from timeit import default_timer
from typing import Callable, Generator

import numpy as np
import torch
from gymnasium import Env
from gymnasium.wrappers import RecordVideo
from numpy import ndarray

from leap_c.controller import CtxType
from leap_c.examples.utils.matplotlib_env import MatplotlibRenderEnv
from leap_c.torch.utils.seed import RngType, mk_seed
from leap_c.utils.gym import seed_env


def episode_rollout(
    policy: Callable[[ndarray], tuple[ndarray, CtxType, dict[str, float] | None]],
    env: Env,
    episodes: int = 1,
    render_episodes: int = 0,
    render_human: bool = False,
    video_folder: str | Path | None = None,
    name_prefix: str | None = None,
    rng: RngType | None = 1042,
) -> Generator[tuple[dict[str, float | bool | list], dict[str, list]], None, None]:
    """Rollout episodes with the given policy.

    Args:
        policy (Callable): The policy to be used for the rollout.
        env (Env): The gym environment.
        episodes (int): The number of episodes to run.
        render_episodes (int): Number of episodes to render. If `0`, no episodes will be rendered.
        render_human (bool): If `True`, render the environment in human mode.
            The environment render mode should then also be human render mode.
            Can not be `True` if `video_path` is provided.
        video_folder (str | Path, optional): The environment is rendered and saved as a
            video in this folder. Can not be set if `render_human` is `True`.
        name_prefix (str, optional): The prefix for the video file names. Must be provided if
            `video_folder` is provided.
        rng (RngType, optional): The random number generator or seed for seeding the environment.

    Yields:
        The first dictionary containing the information about the rollout, at least containing the
        following keys:
         - `"score"`: The cumulative reward of the episode,
         - `"length"`: The length of the episode,
         - `"terminated"`: Whether it terminated,
         - `"truncated"`: Whether it truncated,
         - `"inference_time"`: The average inference time of the policy per step.
        The second dictionary containing statistics returned by the policy.
    """

    def render_trigger(episode_id: int) -> bool:
        return episode_id < render_episodes

    if (
        render_episodes > 0
        and env.render_mode not in (None, "human", "ansi")
        and video_folder is not None
    ):
        if render_human:
            raise ValueError("`render_human` and `video_folder` can not be set at the same time.")
        if name_prefix is None:
            raise ValueError("`name_prefix` must be set if `video_folder` is set.")

        env = RecordVideo(
            env, video_folder, name_prefix=name_prefix, episode_trigger=render_trigger
        )

    rng = np.random.default_rng(rng)
    first_episode = True

    with torch.inference_mode():
        for episode in range(episodes):
            policy_stats = defaultdict(list)
            episode_stats = defaultdict(list)

            if first_episode:
                o, _ = seed_env(env, mk_seed(rng))
                first_episode = False
            else:
                o, _ = env.reset()

            terminated = False
            truncated = False

            cum_inference_time = 0.0

            while not terminated and not truncated:
                t0 = default_timer()
                a, ctx, stats = policy(o)
                cum_inference_time += default_timer() - t0

                if stats is not None:
                    for key, value in stats.items():
                        policy_stats[key].append(value)

                if isinstance(a, torch.Tensor):
                    a = a.cpu().numpy()

                o_prime, _, terminated, truncated, info = env.step(a)

                if "task" in info:
                    for key, value in info["task"].items():
                        episode_stats[key].append(value)

                if render_trigger(episode):
                    if isinstance(env.unwrapped, MatplotlibRenderEnv):
                        env.unwrapped.set_ctx(ctx)

                    if render_human:
                        env.render()

                o = o_prime

            assert "episode" in info, "The environment did not return episode information."
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
