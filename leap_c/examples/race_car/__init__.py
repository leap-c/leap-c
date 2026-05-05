"""Frenet-frame race-car example (bicycle model + parametric acados OCP + MPC planner).

Adapted from the upstream acados ``race_cars`` Python example, with the
continuous-time Frenet-frame bicycle model, track geometry, and OCP cost /
constraint structure taken from Reiter et al. (2023). See the module
docstrings of ``env``, ``bicycle_model``, ``acados_ocp``, and ``planner``
for full state / action / observation specifications.

References:
----------
- Reiter, R., Nurkanović, A., Frey, J., Diehl, M. (2023).
  "Frenet-Cartesian model representations for automotive obstacle
  avoidance within nonlinear MPC."
  European Journal of Control, Vol. 74, 100847.
  Preprint: https://arxiv.org/abs/2212.13115
  Published: https://www.sciencedirect.com/science/article/pii/S0947358023000766
- Upstream code: ``external/acados/examples/acados_python/race_cars/``
"""
