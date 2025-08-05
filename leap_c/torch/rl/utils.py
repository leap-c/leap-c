"""This file contains utility functions that are used in the training loop."""

import torch.nn as nn


def soft_target_update(
    source_net: nn.Module, target_net: nn.Module, tau: float
) -> None:
    """Update the target network parameters using a soft update rule.

    Args:
        source_net: The source network whose parameters are copied to the target network.
        target_net: The target network whose parameters are updated.
        tau: The interpolation parameter for the soft update rule.
    """
    for source_param, target_param in zip(
        source_net.parameters(), target_net.parameters()
    ):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )
