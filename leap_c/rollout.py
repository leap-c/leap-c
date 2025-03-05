"""Contains the necessary functions for validation."""

from collections import defaultdict

from gymnasium import Env
from typing import Callable, Optional
from pathlib import Path
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


import torch


def save_video(
    frames: list,
    video_path: str | Path,
    fps: int,
):
    """Save a video from a list of frames.

    Args:
        frames (list): A list of frames to compose the video.
        video_path (str | Path): The path where the video will be saved.
        fps (int): The frames per second of the video.
    """
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(str(video_path), codec="libx264")


def episode_rollout(
    policy: Callable,
    env: Env,
    render_human: bool = False,
    video_path: Optional[str | Path] = None,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Rollout an episode and returns the cumulative reward.

    Args:
        policy (Callable): The policy to be used for the rollout.
        env (Env): The gym environment.
        val_cfg (ValConfig): The validation configuration.
        render_human (bool): If True, render the environment should be in human render
            mode. Can not be true if video_path is set.
        video_path (Optional[str | Path]): The environment is rendered and saved as a
            video to this path. Can not be set if render_human is True.

    Returns:
        A dictionary containing the information about the rollout, at containing the
        keys "score", "length", "terminated", and "truncated" and a dictionary of
        policy statistics.
    """
    if render_human and video_path is not None:
        raise ValueError("render_human and video_path can not be set at the same time.")

    render = render_human or video_path is not None

    score = 0
    count = 0
    policy_stats = defaultdict(list)
    o, _ = env.reset()

    frames = []
    terminated = False
    truncated = False

    with torch.no_grad():
        while not terminated and not truncated:
            a, stats = policy(o)

            if stats is not None:
                for key, value in stats.items():
                    policy_stats[key].append(value)

            if isinstance(a, torch.Tensor):
                a = a.cpu().numpy()

            o_prime, r, terminated, truncated, _ = env.step(a)

            if render:
                frame = env.render()

                if video_path is not None:
                    frames.append(frame)

            score += r  # type: ignore
            o = o_prime
            count += 1

    if render_human:
        env.close()

    if video_path is not None:  # human mode does not return frames
        try:
            render_fps = env.metadata["render_fps"]
        except KeyError:
            raise ValueError("The environment does not have a render_fps attribute.")

        save_video(frames, video_path, render_fps)

    rollout_stats = {
        "score": score,
        "length": count,
        "terminated": terminated,
        "truncated": truncated,
    }

    return rollout_stats, policy_stats
