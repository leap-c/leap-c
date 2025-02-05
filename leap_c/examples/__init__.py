from mujoco.task import HalfCheetahTask


def create_task(name: str):
    if name == "half_cheetah":
        return HalfCheetahTask()
    else:
        raise ValueError(f"Unknown task: {name}")
