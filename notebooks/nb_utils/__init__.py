"""Shared helpers for the leap-c tutorial notebooks.

Code that is *taught* lives inline in exactly one notebook and has a synced
copy here so the other notebooks can import it instead of repeating the
setup (reciprocal NOTE headers mark the pairs): the heating OCP builder is
taught in ``getting_started/02_from_acados_to_diff_mpc.py``, the
``HeatingPlanner`` in ``getting_started/06_planner_interface.py``. Everything
else (data profiles, the comfort-band OCP, the ``HouseEnv``, plot helpers,
the MSD and prosumer builders) lives only here.
"""
