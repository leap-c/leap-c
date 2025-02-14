"""Prototyping for testing the PointMassMPC and PointMassEnv classes."""

import numpy as np
from leap_c.examples.pointmass.mpc import PointMassMPC
from leap_c.examples.pointmass.env import PointMassEnv
from leap_c.registry import create_task, create_default_cfg, create_trainer


from argparse import ArgumentParser
from dataclasses import asdict
import datetime
from pathlib import Path
import yaml

import leap_c.examples  # noqa: F401
import leap_c.rl  # noqa: F401
from leap_c.registry import create_task, create_default_cfg, create_trainer
from leap_c.trainer import BaseConfig


def run_test_pointmass_functions(mpc: PointMassMPC):
    s = np.array([1.0, 0.0, 0.0, 0.0])
    a = np.array([0.0, 0.0])

    _ = mpc.policy(state=s, p_global=None)[0]
    _ = mpc.state_value(state=s, p_global=None)[0]
    _ = mpc.state_action_value(state=s, action=a, p_global=None)[0]


def run_closed_loop_test(mpc: PointMassMPC, env: PointMassEnv, n_iter: int = int(2e2)):
    s, _ = env.reset(seed=0)
    for _ in range(n_iter):
        a = mpc.policy(state=s, p_global=None)[0]
        s, _, _, _, _ = env.step(a)

    assert np.linalg.norm(s) < 1e-1


def run_closed_loop(
    mpc: PointMassMPC,
    env: PointMassEnv,
    dt: float | None = None,
    n_iter: int = int(2e2),
):
    s, _ = env.reset()

    S = np.zeros((n_iter, 4))
    S[0, :] = s
    A = np.zeros((n_iter, 2))
    for i in range(n_iter - 1):
        A[i, :] = mpc.policy(state=S[i, :], p_global=None)[0]
        S[i + 1, :], _, _, _, _ = env.step(A[i, :])

    plot_data = np.hstack([S, A])
    return plot_data


if __name__ == "__main__":
    trainer = create_trainer(
        name="sac_fou",
        task=create_task("point_mass"),
        output_path="output/videos",
        device="cpu",
        cfg=create_default_cfg("sac_fou"),
    )

    trainer.validate()
