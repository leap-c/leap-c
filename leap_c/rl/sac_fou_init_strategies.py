from leap_c.rl.sac_fou import SACFOPTrainer


class SACFOUTrainerPreviousInit(SACFOPTrainer):
    """SAC-FO-U but always initializing the MPC with the solution from the previous call."""
