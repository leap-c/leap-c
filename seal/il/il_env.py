
# TODO (Jasper): Move this to imitation learning library.
#     def npy_path(self, output_path: Path, name: str):
#         """Declares the path where the numpy files are stored.
# 
#         Args:
#             output_path: The root path where the numpy files are stored.
#             name: The name of the data.
# 
#         Returns:
#             The path where the numpy files are stored.
#         """
#         return output_path / f"{name}_{self.episode_id}.npy"
# 
#     def write_npy(self, output_path: Path):
#         """Write the current episode to a numpy file.
# 
#         Args:
#             output_path: The path where the numpy files are stored.
#         """
#         np.save(self.npy_path(output_path, "state"), self.x_his)
#         np.save(self.npy_path(output_path, "control"), self.u_his)
# 
#     def compare_npy(self, comparison_path: Path) -> dict[str, float | np.ndarray]:
#         """Compare the current episode with the reference episode.
# 
#         Args:
#             comparison_path: The root path where the numpy files are stored.
# 
#         Returns:
#             The MSE stats generated from the comparison.
#         """
#         x_ref = np.load(self.npy_path(comparison_path, "state"))
#         u_ref = np.load(self.npy_path(comparison_path, "control"))
#         x_his = np.stack(self.x_his)
#         u_his = np.stack(self.u_his)
#         assert x_his.shape == x_ref.shape and u_his.shape == u_ref.shape
# 
#         mse_x = np.mean((self.x_his - x_ref) ** 2, axis=0)
#         mse_u = np.mean((self.u_his - u_ref) ** 2, axis=0)
# 
#         return {"mse_x": mse_x, "mse_u": mse_u}


def evaluate_policy(
    policy_fn: Callable[[MPCInput], np.ndarray],
    env: EvalEnv,
    num_episodes: int | None = None,
    seed: int = 0,
    comparison_dir: Path | None = None,
    store_dir: Path | None = None,
) -> pd.DataFrame:
    """Closed-loop evaluation of a policy in terms of costs, constraint violations
    and approximation error to the original planner.

    Args:
        policy_fn: The function that takes the mpc_input and returns the mpc_output.
        env: The environment that is used to evaluate the planner.
        num_episodes: The number of episodes to evaluate the planner. If comparison_dir
            is given, the number of episodes is taken from the comparison_dir.
        seed: The seed used to make the results reproducible. Is used at the first
            reset.
        comparison_dir: The directory of trajectories from a different planner to
            evaluate against.
        store_dir: The directory for storing trajectories generated by the current
            planner.

    Returns:
        The statistics of the evaluation.
    """
    stats = []

    if comparison_dir is not None:
        assert num_episodes is None
        num_episodes = len(list(comparison_dir.glob("state_*.npy")))
    elif hasattr(env, "num_episodes"):
        num_episodes = env.num_episodes  # type: ignore

    assert num_episodes is not None

    for _ in range(num_episodes):
        episode_stats = defaultdict(list)
        episode_return = 0
        episode_length = 0

        # TODO Jasper: Check this again!
        mpc_input, _ = env.reset(seed=seed)
        terminal = False
        truncated = False

        while not terminal and not truncated:
            action = policy_fn(mpc_input)
            mpc_input, cost, terminal, truncated, info = env.step(action)
            episode_return += cost
            episode_length += 1

            for key, value in info.items():
                episode_stats[key].append(value)

            import pdb

            pdb.set_trace()

        final_stats = {}

        # calculate the mean stats ignoring nans
        for key, value in episode_stats.items():
            final_stats[key] = np.nanmean(value)

        if store_dir is not None:
            env.write_npy(store_dir)

        if comparison_dir is not None:
            comp_stats = env.compare_npy(comparison_dir)
            # report each dimension separately
            for key, value in comp_stats.items():
                if isinstance(value, float):
                    final_stats[key] = value
                    continue
                # average over all dimensions
                final_stats[key] = np.mean(value)
                for idx in range(len(value)):  # type: ignore
                    final_stats[key + f"_{idx}"] = value[idx]  # type: ignore

        final_stats["return"] = episode_return
        final_stats["length"] = episode_length
        final_stats["terminal"] = terminal
        final_stats["truncated"] = truncated
        final_stats["episode_id"] = env.episode_id

        stats.append(final_stats)

    return pd.DataFrame(stats)

    @property
    def episode_id(self) -> str:
        return f"episode_{self.episode_idx}"
