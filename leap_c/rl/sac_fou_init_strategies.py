from leap_c.rl.sac_fou import SACFOUTrainer


class SACFOUTrainerPreviousInit(SACFOUTrainer):
    """SAC-FO-U but always initializing the MPC with the solution from the previous call."""
